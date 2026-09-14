# Phase 3: Service Daemonization & Systemd Lifecycle Management

## 1. Architectural Scope
Phase 3 establishes an autonomous, self-healing background system daemon on Arch Linux using systemd user units and journald telemetry.

## 2. Systemd User Service
- Unit path: `~/.config/systemd/user/mcp-rag-watcher.service`
- Supervisor: `Restart=on-failure`, `RestartSec=5s`
- Lingering support: `loginctl enable-linger $USER`
- Standard logging to journald / stderr.

## 3. CLI Service Management
- `filesystem-rag-mcp service install`: Generates and links systemd user unit, reloads daemon, enables service.
- `filesystem-rag-mcp service start / stop / restart`: Controls user unit via `systemctl --user`.
- `filesystem-rag-mcp service status`: Queries active unit status.

## 4. Concurrency Governance
- Enforce non-blocking concurrent reads and immediate write transactions.
- Graceful signal handling for SIGINT / SIGTERM.
