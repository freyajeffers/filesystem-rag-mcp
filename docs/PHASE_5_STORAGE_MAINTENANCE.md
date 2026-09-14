# Phase 5: Automated Storage Maintenance & Operational Runbooks

## 1. Architectural Scope
Phase 5 implements automated database maintenance, index compaction routines, scheduled systemd user timers, and operational disaster recovery runbooks.

## 2. Storage Maintenance Primitives
- Incremental vacuuming and page reclamation.
- Fulltext index segment optimization to merge b-trees and restore peak search speed.
- Integrity auditing (`integrity_check`, `foreign_key_check`).

## 3. Administrative MCP Tool
Expose `optimize_database` tool on the MCP server reporting before and after storage metrics.

## 4. Systemd Maintenance Timer
- Timer unit: `~/.config/systemd/user/mcp-rag-maintenance.timer` running weekly.
- Service unit: `~/.config/systemd/user/mcp-rag-maintenance.service`.

## 5. Operational Runbooks
- Full Index Rebuild Runbook
- Storage Backup & Restoration
- Model Cache Refresh
