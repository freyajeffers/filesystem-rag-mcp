import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def run_live_mcp_tests():
    print("=" * 70)
    print("STARTING REAL-WORLD MCP CLIENT END-TO-END VALIDATION")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        workspace = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        workspace.mkdir()
        data_dir.mkdir()

        # Initialize a real git repo for git_search and graph tests
        subprocess.run(
            ["git", "init", "-b", "main", str(workspace)], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(workspace), "config", "user.name", "Test User"], check=True
        )
        subprocess.run(
            ["git", "-C", str(workspace), "config", "user.email", "test@example.com"], check=True
        )

        # 1. Markdown document with hierarchical headings
        (workspace / "system_architecture.md").write_text(
            """# Platform Architecture
## Overview
The platform provides a decentralized compute fabric.

## Storage Subsystem
### Distributed Vector DB
Vector embeddings are indexed with HNSW and scalar quantization.

### SQLite Full-Text Storage
Lexical search utilizes SQLite FTS5 with BM25 ranking.
"""
        )

        # 2. Python source file with classes and functions
        (workspace / "engine.py").write_text(
            """class ComputeEngine:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.active = True

    def execute_task(self, task_name: str) -> bool:
        print(f"Executing {task_name} on {self.node_id}")
        return True

def initialize_cluster(size: int = 5):
    return [ComputeEngine(f"node-{i}") for i in range(size)]
"""
        )

        # 3. Python file referencing the engine
        (workspace / "main.py").write_text(
            """from engine import ComputeEngine, initialize_cluster

def main():
    cluster = initialize_cluster(3)
    for node in cluster:
        node.execute_task("warmup")

if __name__ == "__main__":
    main()
"""
        )

        # 4. CSV structured data
        (workspace / "metrics.csv").write_text(
            "timestamp,node_id,cpu_pct,mem_mb\n"
            "2026-09-01T00:00:00,node-0,12.5,512\n"
            "2026-09-01T00:01:00,node-1,88.2,1024\n"
            "2026-09-01T00:02:00,node-0,14.1,520\n"
        )

        # 5. JSON data
        (workspace / "config.json").write_text(
            json.dumps(
                {
                    "cluster_name": "alpha-prod",
                    "nodes": [
                        {"id": "node-0", "role": "leader", "capacity": 100},
                        {"id": "node-1", "role": "worker", "capacity": 50},
                    ],
                },
                indent=2,
            )
        )

        # 6. SQLite database
        db_path = workspace / "state.db"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE services (id INTEGER PRIMARY KEY, name TEXT, port INTEGER, active INTEGER);"
        )
        cur.execute("INSERT INTO services VALUES (1, 'api-gateway', 8080, 1);")
        cur.execute("INSERT INTO services VALUES (2, 'auth-service', 8443, 1);")
        cur.execute("INSERT INTO services VALUES (3, 'search-service', 9090, 0);")
        conn.commit()
        conn.close()

        # Initial git commit
        subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "commit",
                "-m",
                "feat: initial commit of platform source and docs",
            ],
            check=True,
        )

        # Start MCP stdio server
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "filesystem_rag_mcp.cli",
                "--transport",
                "stdio",
                "--root-dir",
                str(workspace),
                "--data-dir",
                str(data_dir),
                "--log-level",
                "WARNING",
            ],
            env={**os.environ, "FSRAG_PROFILE": "codebase", "FSRAG_OFFLINE": "true"},
        )

        async with (
            stdio_client(server_params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            init_res = await session.initialize()
            print(
                f"[1/12] Client Handshake OK: {init_res.server_info.name} v{init_res.server_info.version}"
            )

            # Test 1: ping
            ping_out = json.loads((await session.call_tool("ping", {})).content[0].text)
            assert ping_out["success"] is True
            assert ping_out["profile"] == "codebase"
            print("  [OK] ping tool verified (status: healthy)")

            # Test 2: reindex_directory
            reindex_out = json.loads(
                (await session.call_tool("reindex_directory", {"full_rebuild": True}))
                .content[0]
                .text
            )
            assert reindex_out["files_indexed"] >= 5
            print(
                f"  [OK] reindex_directory indexed {reindex_out['files_indexed']} files, {reindex_out['chunks_indexed']} chunks"
            )

            # Test 3: get_index_stats and status
            stats_out = json.loads((await session.call_tool("get_index_stats", {})).content[0].text)
            assert stats_out["quick_index"]["text_chunk_count"] >= 5
            print(
                f"  [OK] get_index_stats returned {stats_out['quick_index']['text_chunk_count']} text chunks in store"
            )

            # Test 4: search with breadcrumbs
            search_out = json.loads(
                (await session.call_tool("search", {"query": "HNSW vector embeddings", "top_k": 3}))
                .content[0]
                .text
            )
            assert search_out["success"] is True
            assert len(search_out["results"]) > 0
            top_hit = search_out["results"][0]
            print(
                f"  [OK] search hit: {top_hit['rel_path']} (score: {top_hit['score']:.4f}, breadcrumb: {top_hit.get('breadcrumbs')})"
            )

            # Test 5: deep_search and pack_context
            deep_out = json.loads(
                (
                    await session.call_tool(
                        "deep_search",
                        {"query": "ComputeEngine initialize_cluster", "top_k_per_subquery": 5},
                    )
                )
                .content[0]
                .text
            )
            assert deep_out["success"] is True
            assert len(deep_out["merged_chunks"]) > 0
            print(f"  [OK] deep_search returned {len(deep_out['merged_chunks'])} expanded hits")

            pack_out = json.loads(
                (
                    await session.call_tool(
                        "pack_context", {"query": "ComputeEngine", "max_tokens": 1000}
                    )
                )
                .content[0]
                .text
            )
            assert pack_out["success"] is True
            assert len(pack_out["markdown"]) > 50
            print(
                f"  [OK] pack_context assembled {pack_out['chunks_included']} chunks ({pack_out['unique_files_included']} files) within {pack_out['estimated_tokens']} estimated tokens"
            )

            # Test 6: search_symbols and find_symbol_references
            symbols_out = json.loads(
                (await session.call_tool("search_symbols", {"name": "ComputeEngine"}))
                .content[0]
                .text
            )
            assert symbols_out["success"] is True
            assert any(s["name"] == "ComputeEngine" for s in symbols_out["symbols"])
            print(f"  [OK] search_symbols found: {[s['name'] for s in symbols_out['symbols']]}")

            refs_out = json.loads(
                (
                    await session.call_tool(
                        "find_symbol_references", {"symbol_name": "ComputeEngine"}
                    )
                )
                .content[0]
                .text
            )
            assert refs_out["success"] is True
            assert refs_out["total_references"] >= 2
            print(
                f"  [OK] find_symbol_references located {refs_out['total_references']} references across files"
            )

            # Test 7: grep_search
            grep_out = json.loads(
                (await session.call_tool("grep_search", {"pattern": "execute_task"}))
                .content[0]
                .text
            )
            assert grep_out["success"] is True
            assert grep_out["total_matches"] >= 2
            print(f"  [OK] grep_search found {grep_out['total_matches']} occurrences")

            # Test 8: get_corpus_graph and git_search
            graph_out = json.loads(
                (await session.call_tool("get_corpus_graph", {})).content[0].text
            )
            assert graph_out["success"] is True
            print(
                f"  [OK] get_corpus_graph mapped {graph_out['total_files']} files and {graph_out['total_dependencies']} dependencies"
            )

            git_out = json.loads(
                (await session.call_tool("git_search", {"query": "initial commit"})).content[0].text
            )
            assert git_out["success"] is True
            assert len(git_out["commits"]) > 0
            print(
                f"  [OK] git_search found commit: {git_out['commits'][0]['commit'][:7]} - '{git_out['commits'][0]['subject']}'"
            )

            # Test 9: targeted fetchers (JSON, CSV, SQLite)
            csv_data = json.loads(
                (
                    await session.call_tool(
                        "fetch_targeted_data",
                        {
                            "rel_path": "metrics.csv",
                            "filter_col": "node_id",
                            "filter_value": "node-1",
                        },
                    )
                )
                .content[0]
                .text
            )
            assert csv_data["success"] is True
            assert len(csv_data["rows"]) == 1
            assert csv_data["rows"][0]["cpu_pct"] == "88.2"
            print("  [OK] fetch_targeted_data (CSV) verified")

            json_data = json.loads(
                (
                    await session.call_tool(
                        "fetch_targeted_data", {"rel_path": "config.json", "query": "nodes[0].role"}
                    )
                )
                .content[0]
                .text
            )
            assert json_data["success"] is True
            assert json_data["result"] == "leader"
            print(f"  [OK] fetch_targeted_data (JSON) verified: {json_data['result']}")

            sql_data = json.loads(
                (
                    await session.call_tool(
                        "fetch_targeted_data",
                        {
                            "rel_path": "state.db",
                            "query": "SELECT name, port FROM services WHERE active = 1",
                        },
                    )
                )
                .content[0]
                .text
            )
            assert sql_data["success"] is True
            assert len(sql_data["rows"]) == 2
            print(
                f"  [OK] fetch_targeted_data (SQLite) retrieved {len(sql_data['rows'])} rows: {sql_data['rows']}"
            )

            # Test 10: patch_file dry-run and execution
            patch_dry = json.loads(
                (
                    await session.call_tool(
                        "patch_file",
                        {
                            "rel_path": "engine.py",
                            "old_string": "self.active = True",
                            "new_string": "self.active = True\n        self.version = '2.0.0'",
                            "dry_run": True,
                        },
                    )
                )
                .content[0]
                .text
            )
            assert patch_dry["dry_run"] is True
            assert patch_dry["replacements"] == 1
            print("  [OK] patch_file dry run verified")

            patch_exec = json.loads(
                (
                    await session.call_tool(
                        "patch_file",
                        {
                            "rel_path": "engine.py",
                            "old_string": "self.active = True",
                            "new_string": "self.active = True\n        self.version = '2.0.0'",
                            "dry_run": False,
                        },
                    )
                )
                .content[0]
                .text
            )
            assert patch_exec["success"] is True
            assert "self.version = '2.0.0'" in (workspace / "engine.py").read_text()
            print("  [OK] patch_file executed and file modified on disk")

            # Test 11: Security and Agent-readable error schema on path traversal
            err_resp = json.loads(
                (await session.call_tool("read_file", {"rel_path": "../../../etc/passwd"}))
                .content[0]
                .text
            )
            assert err_resp["error"]["code"] == "PATH_TRAVERSAL_BLOCKED"
            print(
                f"  [OK] Security validation verified (PATH_TRAVERSAL_BLOCKED rejected: {err_resp['error']['message']})"
            )

            # Test 12: optimize_database tool
            opt_out = json.loads((await session.call_tool("optimize_database", {})).content[0].text)
            assert opt_out["success"] is True
            assert opt_out["report"]["ok"] is True
            print(
                f"  [OK] optimize_database completed in {opt_out['report']['duration_ms']}ms (reclaimed: {opt_out['report']['reclaimed_bytes']} bytes)"
            )

    print("=" * 70)
    print("ALL 12 REAL-WORLD MCP CLIENT TEST CATEGORIES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_live_mcp_tests())
