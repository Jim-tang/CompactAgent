"""持久化任务队列与后台 worker：认领 pending 任务、调用对应处理器、写回状态与重试退避。

状态机：``pending → running → done``；失败 ``→ failed →``（退避重试）``→ dead``。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from .store import MemoryStore
from .utils import ulid

logger = logging.getLogger(__name__)


class MemoryWorker:
    """后台任务循环：从库里认领任务、执行处理器、按结果置 done / 重试 / 死信。"""

    def __init__(
        self,
        store: MemoryStore,
        config: dict,
        handlers: Optional[dict[str, Callable]] = None,
        on_task_settled: Optional[Callable] = None,
    ) -> None:
        self.store = store
        self.config = config or {}
        self.handlers: dict[str, Callable] = handlers or {}
        self.on_task_settled = on_task_settled  # 可选回调：任务收尾（置 done / 放回 pending / 进死信）之后调用
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self.max_retries = int(self.config.get("task_max_retries", 3))
        self.backoff_seconds = list(self.config.get("task_backoff_seconds", [5, 15, 45]))
        self.poll_seconds = float(self.config.get("worker_poll_seconds", 1.0))
        self.stale_seconds = int(self.config.get("task_stale_seconds", 600))

    async def start(self) -> None:
        """把卡死的 running 任务重置为 pending，然后启动主循环 task。"""
        self._stopping.clear()
        recovered = await asyncio.to_thread(
            self.store.recover_stale_tasks, self.stale_seconds
        )
        if recovered:
            logger.info("已把 %d 个卡死的 running 任务重置为 pending", recovered)
        self._task = asyncio.create_task(self._run(), name="memory-worker")

    async def stop(self, flush: bool = True) -> None:
        """停止主循环；flush=True 时先等队列排空（最多 10 秒）再退出。"""
        if self._task is None:
            self._stopping.set()
            return
        if flush:
            # 必须先等队列排空再置 _stopping：主循环的 while 条件读的就是它，
            # 先置位会让循环立刻退出，wait_idle() 就等成一个空转的 10 秒超时，
            # 最后一条入队任务永远停在 pending。
            try:
                await self.wait_idle(timeout=10.0)
            except Exception:  # noqa: BLE001 - 关闭路径不抛
                pass
        self._stopping.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    async def wait_idle(self, timeout: float = 30.0) -> bool:
        """轮询到任务表清空且认领过的那一轮彻底跑完为止，超时返回 False。"""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            depth = await self.depth()
            if (
                depth.get("pending", 0) == 0
                and depth.get("running", 0) == 0
                and self._idle.is_set()  # 任务置 done 之后还要跑收尾回环
            ):
                return True
            await asyncio.sleep(0.05)
        return False

    async def enqueue(
        self,
        kind: str,
        scope: str,
        payload: dict,
        session_id: Optional[str] = None,
    ) -> str:
        """写入一条 pending 任务，返回 task_id。"""
        task_id = ulid()
        await asyncio.to_thread(
            self.store.enqueue_task, task_id, kind, scope, payload, session_id
        )
        self._idle.clear()
        return task_id

    async def enqueue_extract(
        self,
        session_id: str,
        from_id: int,
        until_id: Optional[int] = None,
        scope: str = "",
    ) -> str:
        """入队一次 ``extract_l1`` 任务，返回 task_id。

        ``until_id`` 只是意向上界，游标取的是任务成功后写入的 ``to_id``，
        这样抽取失败不会推进游标。
        """
        payload: dict[str, Any] = {"session_id": session_id, "from_id": from_id}
        if until_id is not None:
            payload["until_id"] = until_id
        return await self.enqueue(
            kind="extract_l1", scope=scope, payload=payload, session_id=session_id
        )

    async def enqueue_l2(self, trigger: str, scope: str = "") -> str:
        """入队一次 ``consolidate_l2`` 任务，``trigger`` 记录本次的触发原因。"""
        return await self.enqueue(
            kind="consolidate_l2", scope=scope, payload={"trigger": trigger}
        )

    async def depth(self) -> dict[str, int]:
        """统计各状态任务条数，返回 ``{status: count}``。"""
        counts: dict[str, int] = {}
        for status in ("pending", "running", "done", "failed", "dead"):
            value = await asyncio.to_thread(
                self.store.scalar,
                "SELECT COUNT(*) FROM memory_task WHERE status = ?", (status,)
            )
            counts[status] = int(value or 0)
        return counts

    async def _run(self) -> None:
        """主循环：认领一个任务就执行，没有任务则等待 ``poll_seconds`` 后重试，直到停止。"""
        while not self._stopping.is_set():
            claimed = False
            try:
                task = await asyncio.to_thread(self.store.claim_next_task)
                if task is None:
                    self._idle.set()
                    try:
                        await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_seconds)
                    except asyncio.TimeoutError:
                        pass
                    continue
                claimed = True
                # 先清 idle 再执行：wait_idle() 必须一直等到任务真正结束，否则在「已认领、未完成」的窗口里会误判为空闲。
                self._idle.clear()
                await self._handle(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 循环本身绝不因单次异常退出
                logger.exception("worker 循环异常: %s", exc)
                await asyncio.sleep(self.poll_seconds)
            finally:
                # 只有「本轮认领过任务」才需要重新置位：空转分支里 idle 本来就是 set 状态；
                # 若在这里无条件 set，认领后立刻 stop() 会把尚未执行完的那一轮误报为空闲。
                if claimed:
                    self._idle.set()

    async def _handle(self, task: dict) -> None:
        """执行单个任务，处理器全部成功后才置 done。

        handler 返回 dict 时视为 ``payload_patch``，在置 done 的同一次 UPDATE 里补写进 payload；
        抛异常则按退避重试，重试次数用尽置 dead，取消则放回 pending。
        """
        kind = task.get("kind", "")
        task_id = task["task_id"]
        handler = self.handlers.get(kind)
        if handler is None:
            await asyncio.to_thread(
                self.store.finish_task, task_id, "dead", f"未知任务类型: {kind}"
            )
            logger.error("未知任务类型 %s（task_id=%s）", kind, task_id)
            return

        import json as _json

        try:
            payload = _json.loads(task.get("payload_json") or "{}")
        except (ValueError, TypeError):
            payload = {}

        patch: Optional[dict] = None
        try:
            result = await handler(payload, task)
            if isinstance(result, dict):
                patch = result
        except asyncio.CancelledError:
            # 进程被停：把任务放回 pending，别让它永远卡在 running
            await asyncio.to_thread(
                self.store.finish_task, task_id, "pending", "worker 被取消"
            )
            raise
        except Exception as exc:
            attempts = int(task.get("attempts", 0))
            if attempts >= self.max_retries:
                await asyncio.to_thread(
                    self.store.finish_task, task_id, "dead", str(exc)[:500]
                )
                logger.error("任务进入死信（attempts=%d）: %s", attempts, exc)
            else:
                delay = self.backoff_seconds[min(attempts - 1, len(self.backoff_seconds) - 1)]
                retry_at = (datetime.now() + timedelta(seconds=delay)).isoformat()
                await asyncio.to_thread(
                    self.store.finish_task,
                    task_id, "pending", str(exc)[:500], next_retry_at=retry_at
                )
                logger.warning("任务失败将重试（%ds 后）: %s", delay, exc)
            return

        await asyncio.to_thread(
            self.store.finish_task, task_id, "done", payload_patch=patch
        )
        await self._settle(task, patch)

    async def _settle(self, task: dict, patch: Optional[dict]) -> None:
        """把已收尾的任务回传给 ``on_task_settled``，回调异常只记日志。

        回传内容补上 ``changed_scenes``：L2 处理器的 ``payload_patch`` 只带这个字段，
        由 worker 解成 int，回调方不必再解析 payload 结构。
        """
        if self.on_task_settled is None:
            return
        settled = dict(task)
        try:
            settled["changed_scenes"] = int((patch or {}).get("changed_scenes") or 0)
        except (TypeError, ValueError):
            settled["changed_scenes"] = 0
        try:
            await self.on_task_settled(settled)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 编排失败不影响任务结果
            logger.exception("任务收尾回环失败（task_id=%s）: %s", task.get("task_id"), exc)
