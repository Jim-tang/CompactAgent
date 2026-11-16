from __future__ import annotations

import secrets
from datetime import datetime


class SessionManager:
    """会话 ID 的唯一来源。"""

    def __init__(self) -> None:
        self.session_id = self.create_session_id()

    @staticmethod
    def create_session_id() -> str:
        """会话 ID：s_<YYYYmmddHHMMSS>_<6 位随机小写字母数字>（§4.3）。"""
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = "".join(secrets.choice(chars) for _ in range(6))
        return f"s_{stamp}_{suffix}"

    def get_id(self) -> str:
        return self.session_id

    def new_session(self) -> str:
        """生成并持有新的 session_id。"""
        self.session_id = self.create_session_id()
        return self.session_id
