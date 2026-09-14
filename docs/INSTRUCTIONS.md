# System Setup, Operational Workflows & Runbooks

## Local Filesystem RAG MCP Server

### 1. Prerequisites & Environment Setup

This section outlines the setup procedure for Arch Linux workstations running Python 3.12+.

- **Package Manager**: The repository uses `uv` for fast, deterministic dependency resolution and virtual environment management.
- **Virtual environment target directory**: `~/.local/share/mcp-rag/venv`
- **Synchronize dependencies**: Run `uv sync` to install pinned packages.
- **Directory Layout**: Verify or create the default personal notes vault directory:
  ```bash
  mkdir -p ~/.local/share/notes
  ```

### 2. Operational CLI Commands

- **Run Diagnostics**:
  ```bash
  python -m filesystem_rag_mcp.cli doctor
  ```
- **Synchronous Indexing**:
  ```bash
  python -m filesystem_rag_mcp.cli index --thorough
  ```
- **Search**:
  ```bash
  python -m filesystem_rag_mcp.cli search "query text" --rerank
  ```
- **Benchmark Evaluation**:
  ```bash
  python -m filesystem_rag_mcp.cli benchmark
  ```
- **Service Management**:
  ```bash
  python -m filesystem_rag_mcp.cli service install
  python -m filesystem_rag_mcp.cli service start
  python -m filesystem_rag_mcp.cli service status
  ```
- **Storage Maintenance**:
  ```bash
  python -m filesystem_rag_mcp.cli optimize
  ```

### 3. Client Integration Configurations

#### Claude Desktop Configuration (`~/.config/Claude/claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "filesystem-rag": {
      "command": "/home/user/.local/share/mcp-rag/venv/bin/python",
      "args": ["-m", "filesystem_rag_mcp.server"],
      "env": {
        "FSRAG_PROFILE": "notes"
      }
    }
  }
}
```

#### Zed Editor Configuration (`~/.config/zed/settings.json`)
```json
{
  "context_servers": {
    "filesystem-rag": {
      "command": {
        "path": "/home/user/.local/share/mcp-rag/venv/bin/python",
        "args": ["-m", "filesystem_rag_mcp.server"]
      }
    }
  }
}
```

### 4. Operational Runbooks & Troubleshooting

#### Resolving Database Locks
- **Symptom**: Operations fail with "database is locked" or busy timeout exhaustion.
- **Resolution**: Verify that store is using WAL mode. Check for stale processes holding locks using `fuser` or `lsof` on database files and terminate orphaned instances cleanly.

#### Inotify Watch Limit Exhaustion
- **Symptom**: Watcher daemon fails with error indicating inotify watch limits reached.
- **Resolution**: Increase system inotify watch limits in `/etc/sysctl.d/99-sysctl.conf` by adding `fs.inotify.max_user_watches = 524288` and reloading with `sudo sysctl -p`.

#### Full Index Rebuild Procedure
1. Stop background watcher service: `systemctl --user stop mcp-rag-watcher.service`
2. Execute complete rebuild: `python -m filesystem_rag_mcp.cli index --thorough` (or with `--reindex-all`)
3. Restart service: `systemctl --user start mcp-rag-watcher.service`
4. Verify index stats: `python -m filesystem_rag_mcp.cli stats`
