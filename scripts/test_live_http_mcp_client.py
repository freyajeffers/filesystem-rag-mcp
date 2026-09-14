import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client


async def wait_for_port(port: int, timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            await asyncio.sleep(0.1)
    return False


async def test_streamable_http():
    print("\n" + "=" * 60)
    print("TESTING STREAMABLE HTTP TRANSPORT (PORT 8912)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        workspace = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        workspace.mkdir()
        data_dir.mkdir()

        (workspace / "stream_guide.md").write_text(
            "# Streaming Protocol Architecture\n\nStreamable HTTP provides multiplexed bidirectional message channels over standard HTTP chunked transfer."
        )

        port = 8912
        env = {
            **os.environ,
            "FSRAG_OFFLINE": "true",
            "FSRAG_PROFILE": "codebase",
        }
        cmd = [
            sys.executable,
            "-m",
            "filesystem_rag_mcp.cli",
            "--transport",
            "streamable-http",
            "--port",
            str(port),
            "--no-auth",
            "--root-dir",
            str(workspace),
            "--data-dir",
            str(data_dir),
            "--log-level",
            "WARNING",
        ]

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            # Wait for server to bind port
            bound = await wait_for_port(port)
            if not bound:
                _, stderr = proc.communicate(timeout=1)
                raise RuntimeError(f"Server failed to bind port {port}:\n{stderr.decode()}")
            url = f"http://127.0.0.1:{port}/mcp"

            print(f"Connecting client to Streamable HTTP endpoint: {url} ...")
            async with (
                streamable_http_client(url) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                init_res = await session.initialize()
                print(
                    f"  [OK] Handshake Initialized: {init_res.server_info.name} v{init_res.server_info.version}"
                )

                # Tools list
                tools = await session.list_tools()
                print(f"  [OK] List tools discovered {len(tools.tools)} tools")

                # Ping
                ping_raw = await session.call_tool("ping", {})
                ping_data = json.loads(ping_raw.content[0].text)
                assert ping_data["success"] is True
                print("  [OK] Tool ping response:", ping_data)

                # Reindex
                reindex_raw = await session.call_tool("reindex_directory", {"full_rebuild": True})
                reindex_data = json.loads(reindex_raw.content[0].text)
                assert reindex_data["files_indexed"] >= 1
                print(
                    f"  [OK] Reindex indexed {reindex_data['files_indexed']} files, {reindex_data['chunks_indexed']} chunks"
                )

                # Search
                search_raw = await session.call_tool(
                    "search", {"query": "Streaming Protocol", "top_k": 3}
                )
                search_data = json.loads(search_raw.content[0].text)
                assert search_data["success"] is True
                assert len(search_data["results"]) > 0
                hit = search_data["results"][0]
                print(
                    f"  [OK] Search hit: {hit['rel_path']} (score: {hit['score']:.4f}, breadcrumb: {hit.get('breadcrumbs')})"
                )

                # Optimize
                opt_raw = await session.call_tool("optimize_database", {})
                opt_data = json.loads(opt_raw.content[0].text)
                assert opt_data["success"] is True
                print(f"  [OK] Storage optimization: ok={opt_data['report']['ok']}")

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


async def test_sse_transport():
    print("\n" + "=" * 60)
    print("TESTING SSE (SERVER-SENT EVENTS) TRANSPORT (PORT 8913)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        workspace = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        workspace.mkdir()
        data_dir.mkdir()

        (workspace / "sse_notes.md").write_text(
            "# Server-Sent Events Guide\n\nSSE enables unidirectional real-time event pushing from server to client with automatic reconnection."
        )

        port = 8913
        env = {
            **os.environ,
            "FSRAG_OFFLINE": "true",
            "FSRAG_PROFILE": "notes",
        }
        cmd = [
            sys.executable,
            "-m",
            "filesystem_rag_mcp.cli",
            "--transport",
            "sse",
            "--port",
            str(port),
            "--no-auth",
            "--root-dir",
            str(workspace),
            "--data-dir",
            str(data_dir),
            "--log-level",
            "WARNING",
        ]

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            # Wait for server to bind port
            bound = await wait_for_port(port)
            if not bound:
                _, stderr = proc.communicate(timeout=1)
                raise RuntimeError(f"Server failed to bind port {port}:\n{stderr.decode()}")
            url = f"http://127.0.0.1:{port}/sse"

            print(f"Connecting client to SSE endpoint: {url} ...")
            async with (
                sse_client(url) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                init_res = await session.initialize()
                print(
                    f"  [OK] SSE Handshake Initialized: {init_res.server_info.name} v{init_res.server_info.version}"
                )

                # Ping
                ping_raw = await session.call_tool("ping", {})
                ping_data = json.loads(ping_raw.content[0].text)
                assert ping_data["success"] is True
                assert ping_data["profile"] == "notes"
                print("  [OK] SSE Tool ping response:", ping_data)

                # Reindex
                reindex_raw = await session.call_tool("reindex_directory", {"full_rebuild": True})
                reindex_data = json.loads(reindex_raw.content[0].text)
                assert reindex_data["files_indexed"] >= 1
                print(
                    f"  [OK] Reindex indexed {reindex_data['files_indexed']} files, {reindex_data['chunks_indexed']} chunks"
                )

                # Search Notes
                search_raw = await session.call_tool(
                    "search_notes", {"query": "Server-Sent Events", "top_k": 3}
                )
                search_data = json.loads(search_raw.content[0].text)
                assert search_data["success"] is True
                assert len(search_data["results"]) > 0
                hit = search_data["results"][0]
                print(
                    f"  [OK] Search Notes hit: {hit['rel_path']} (score: {hit['score']:.4f}, breadcrumb: {hit.get('breadcrumbs')})"
                )

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


async def main():
    await test_streamable_http()
    await test_sse_transport()
    print("\n" + "=" * 60)
    print("ALL STREAMABLE HTTP AND SSE REAL-WORLD CLIENT TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
