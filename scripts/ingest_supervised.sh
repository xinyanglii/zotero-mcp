#!/usr/bin/env bash
# Supervisor that re-launches zotero_mcp.ingest after clean exits (CB trips)
# or crashes, sleeping WEBDAV_PROBE_INTERVAL seconds between attempts. Stops
# when SQLite papers.count >= ZOTERO_TOTAL or a sentinel file is touched.
set -u

LOG="${LOG:-$HOME/workspace/claude-workspace/logs/kg_ingest_supervised_$(date +%Y%m%d).log}"
SENTINEL="${SENTINEL:-/tmp/zotero_ingest_stop}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-120}"
WORKERS="${WORKERS:-4}"

# systemd-run --user doesn't inherit $PATH for conda; use full path.
PY="${PY:-$HOME/miniconda3/envs/claude/bin/python}"
if [[ ! -x "$PY" ]]; then
    # fall back to conda-activate flow for interactive use
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate claude
    PY=python
fi
cd "$HOME/workspace/zotero-mcp"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [supervisor] $*" | tee -a "$LOG"; }

log "supervisor started (pid=$$, log=$LOG, workers=$WORKERS)"
log "sentinel=$SENTINEL (touch this file to stop gracefully)"

attempt=0
while :; do
    attempt=$((attempt+1))
    if [[ -f "$SENTINEL" ]]; then
        log "sentinel present, stopping supervisor"
        exit 0
    fi
    log "=== attempt $attempt: launching ingest ==="
    "$PY" -u -m zotero_mcp.ingest --workers "$WORKERS" >> "$LOG" 2>&1
    rc=$?
    log "ingest exited rc=$rc"
    # Peek latest status breakdown
    last_status=$(grep -E "^.*DONE in.*status breakdown" "$LOG" | tail -1)
    [[ -n "$last_status" ]] && log "$last_status"
    # Check done count
    done=$("$PY" -c "
import sqlite3
c=sqlite3.connect('$HOME/.cache/zotero-mcp/kg.sqlite')
print(c.execute('SELECT COUNT(*) FROM papers').fetchone()[0])
")
    log "sqlite papers.count=$done"
    log "sleeping ${SLEEP_BETWEEN}s before next attempt"
    sleep "$SLEEP_BETWEEN"
done
