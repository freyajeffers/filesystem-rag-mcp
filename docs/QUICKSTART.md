# Quickstart Guide for filesystem-rag-mcp

This guide helps you set up and connect `filesystem-rag-mcp` with your AI coding agents, editors, and desktop assistants.

---

## 1. Installation

### Recommended: Using `uv`
```bash
git clone https://github.com/freyajeffers/filesystem-rag-mcp.git
cd filesystem-rag-mcp
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Verify Installation
Run the built-in system doctor:
```bash
filesystem-rag-mcp --doctor
```
This inspects directory permissions, Python version, embedding model capability, and installed file format converters.

---

## 2. Fast Client Configuration Snippets

The CLI includes an automatic configuration generator:
```bash
# Generate snippet for Claude Desktop
filesystem-rag-mcp --config-snippet claude --root-dir /path/to/my/project

# Generate snippet for Zed Editor
filesystem-rag-mcp --config-snippet zed --root-dir /path/to/my/project

# Generate snippet for Hermes / Antigravity Agent
filesystem-rag-mcp --config-snippet hermes --root-dir /path/to/my/project

# Generate snippets for all supported clients
filesystem-rag-mcp --config-snippet all --root-dir /path/to/my/project
```

---

## 3. Client Setup Examples

### Claude Desktop (`claude_desktop_config.json`)
Location:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "filesystem-rag": {
      "command": "/path/to/filesystem-rag-mcp/.venv/bin/python",
      "args": [
        "-m",
        "filesystem_rag_mcp.cli",
        "--transport",
        "stdio",
        "--root-dir",
        "/path/to/my/documents"
      ]
    }
  }
}
```

### Hermes Agent (`config.yaml`)
In `~/.hermes/config.yaml`:
```yaml
mcp:
  servers:
    filesystem-rag:
      command: "/path/to/filesystem-rag-mcp/.venv/bin/python"
      args:
        - "-m"
        - "filesystem_rag_mcp.cli"
        - "--transport"
        - "stdio"
        - "--root-dir"
        - "/path/to/my/documents"
```

### Cursor (`cursor.json` or MCP settings)
```json
{
  "mcpServers": {
    "filesystem-rag": {
      "command": "/path/to/filesystem-rag-mcp/.venv/bin/python",
      "args": [
        "-m",
        "filesystem_rag_mcp.cli",
        "--transport",
        "stdio",
        "--root-dir",
        "/path/to/my/documents"
      ]
    }
  }
}
```

---

## 4. Running as a Network Service (Streamable HTTP)

For remote or multi-agent deployments, start the HTTP server:

```bash
# Development (no OAuth required)
filesystem-rag-mcp --transport http --no-auth --host 127.0.0.1 --port 8000 --root-dir /path/to/docs

# Production (with OAuth 2.1 & Dynamic Client Registration)
export FSRAG_OAUTH_SECRET="your-secure-random-token"
filesystem-rag-mcp --transport http --host 0.0.0.0 --port 8000 --root-dir /path/to/docs
```
