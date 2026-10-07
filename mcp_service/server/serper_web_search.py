import os
import httpx
from typing import Annotated
from pydantic import Field
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

SERPER_API_URL = "https://google.serper.dev/search"
mcp = FastMCP("Web Search with Serper")

@mcp.tool()
async def web_search(
    query: Annotated[str, Field(description="要搜索的关键词")],
    num_results: Annotated[int, Field(description="需要返回的结果数量")] = 5
) -> str:
    """使用 Serper API 执行异步搜索，返回相关结果的标题、链接与摘要"""
    mcp_headers = get_http_headers()  # 获取客户端请求的 headers
    serper_api_key = mcp_headers.get('x-api-key') or os.environ.get("SERPER_API_KEY")
    if not serper_api_key:
        return "Error: SERPER_API_KEY not found."

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    try:
        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.post(SERPER_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return f"Request failed: {e}"

    organic = data.get("organic", [])
    if not organic:
        return "No results."

    results = []
    for item in organic[:num_results]:
        results.append(
            f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('snippet')}\n---"
        )
    return f"Search Results for '{query}':\n\n" + "\n".join(results)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="localhost", port=4200, path="/demo", log_level="debug")