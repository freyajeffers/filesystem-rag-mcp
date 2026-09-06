import json
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

from filesystem_rag_mcp.config import Settings
from filesystem_rag_mcp.oauth import MCPFileRAGAuthProvider


@pytest.mark.asyncio
async def test_client_id_metadata_document_resolution(tmp_path: Path):
    settings = Settings(
        root_dir=tmp_path,
        data_dir=tmp_path / ".fsrag",
        auth_required=True,
    )
    provider = MCPFileRAGAuthProvider(settings)

    metadata_payload = {
        "client_name": "Decentralized MCP Agent",
        "redirect_uris": ["http://127.0.0.1:9999/callback"],
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
        "scope": "fs.rag.read",
    }

    async def metadata_endpoint(request):
        return JSONResponse(metadata_payload)

    app = Starlette(routes=[Route("/client-metadata.json", metadata_endpoint)])
    config = uvicorn.Config(app, host="127.0.0.1", port=8976, log_level="warning")
    server = uvicorn.Server(config)

    import asyncio

    task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.05)

    client_id_url = "http://127.0.0.1:8976/client-metadata.json"
    client_info = await provider.get_client(client_id_url)

    server.should_exit = True
    await task

    assert client_info is not None
    assert str(client_info.client_id) == client_id_url
    assert client_info.client_name == "Decentralized MCP Agent"
    assert any(str(uri) == "http://127.0.0.1:9999/callback" for uri in (client_info.redirect_uris or []))
