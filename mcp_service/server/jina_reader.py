import os
import httpx
from typing import Annotated
from pydantic import Field
from fastmcp import FastMCP

mcp = FastMCP("Jina Reader Server")

@mcp.tool()
async def jina_read_url(
    target_url: Annotated[str, Field(description="需要读取的单个url链接")]
) -> str:
    """读取单个网页并提取内容为干净的 Markdown 文本"""
    jina_api_key = os.environ.get("JINA_API_KEY")
    if not jina_api_key:
        return "Error: JINA_API_KEY not found."

    jina_api_url = f"https://r.jina.ai/{target_url}"
    headers = {"Authorization": f"Bearer {jina_api_key}"}

    try:
        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.get(jina_api_url, headers=headers)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        return f"❌ 读取 {jina_api_url} 时出错: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
