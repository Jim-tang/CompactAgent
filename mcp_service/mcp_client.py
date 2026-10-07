import os
import json
import httpx
import asyncio
import requests
from typing import Dict, Any, List, Optional

os.environ["FASTMCP_LOG_ENABLED"] = "false"         # 关闭 FastMCP 的维测日志输出
os.environ["FASTMCP_SHOW_SERVER_BANNER"] = "false"  # 关闭 FastMCP 的初始化横幅显示
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport, StdioTransport


# 创建一个显式禁用代理的 httpx 客户端
def create_no_proxy_client(headers, **kwargs):
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(proxy=None),
        headers=headers
    )

class MCPClient:
    """
    通用单服务 MCP 客户端（支持 Stdio 和 HTTP）
    """
    def __init__(self, server_name: str, server_config: dict):
        self.server_name = server_name
        self.server_config = server_config
        self._client: Optional[Client] = None
        self.init_error: Optional[str] = None
        self._init_client()

    def _init_client(self):
        """根据配置解析对应的传输层"""
        try:
            if self.server_config.get("type") == "streamable-http":
                transport = StreamableHttpTransport(
                    url=self.server_config.get("url"),
                    headers=self.server_config.get("headers"),  # 向服务端传递配置文件中自定义的headers
                    httpx_client_factory=create_no_proxy_client
                )
            elif self.server_config.get("type") == "stdio":
                # StdioTransport 内部包装了 asyncio.create_subprocess_exec，在连接客户端时会自动拉起 MCP 服务的本地进程
                transport = StdioTransport(
                    command=self.server_config["command"],
                    args=self.server_config.get("args", []),
                    env=self.server_config.get("env", {})
                )
            else:
                raise ValueError(f"服务 [{self.server_name}] 配置不合法（缺少 type 或类型不支持）")
            self._client = Client(transport)
        except Exception as e:
            self.init_error = str(e)

    @property
    def client(self) -> Client:
        return self._client


class MCPClientManager:
    """
    MCP 多服务管理器：并发管理配置文件中的所有服务
    """
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.clients: Dict[str, MCPClient] = {}
        self._load_config()

    def _load_config(self):
        """加载配置文件并初始化所有客户端实例"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        mcp_servers = config_data.get("mcpServers", {})
        for name, config in mcp_servers.items():
            self.clients[name] = MCPClient(name, config)

    async def async_list_tools(self) -> tuple[Dict[str, list], Dict[str, str]]:
        """
        异步并发连接所有 MCP 服务，并聚合返回它们提供的所有工具
        返回: (tools_dict, errors_dict)
        """
        mcp_tools = {}
        mcp_errors = {}

        # 定义单个服务的异步获取任务
        async def fetch_tools(service_name: str, mcp_client: MCPClient):
            if mcp_client.init_error:
                return service_name, [], mcp_client.init_error
            try:
                # 使用 async with 管理短连接的生命周期
                async with mcp_client.client:
                    tools = await mcp_client.client.list_tools()
                    result = []
                    for tool in tools:
                        result.append({
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        })
                    return service_name, result, None
            except Exception as e:
                return service_name, [], str(e)

        # 使用 asyncio.gather 并发执行所有服务的请求
        tasks = [fetch_tools(name, client) for name, client in self.clients.items()]
        results = await asyncio.gather(*tasks)

        for name, tools, error in results:
            mcp_tools[name] = tools
            if not tools:
                mcp_errors[name] = error
        return mcp_tools, mcp_errors

    async def async_call_tool(self, server_name: str, tool_name: str, **kwargs) -> str:
        """
        指定某个具体的服务，调用其下的工具
        """
        if server_name not in self.clients:
            raise ValueError(f"未找到名为 '{server_name}' 的服务")

        mcp_client = self.clients[server_name]
        async with mcp_client.client:
            result = await mcp_client.client.call_tool(tool_name, arguments=kwargs)
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return ""

    # --- 同步包装方法（方便在同步环境或快速测试中使用） ---
    def list_tools(self) -> tuple[Dict[str, list], Dict[str, str]]:
        """同步获取所有服务工具"""
        return asyncio.run(self.async_list_tools())

    def call_tool(self, server_name: str, tool_name: str, **kwargs) -> str:
        """同步调用指定服务的工具"""
        return asyncio.run(self.async_call_tool(server_name, tool_name, **kwargs))