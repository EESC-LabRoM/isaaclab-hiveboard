#!/usr/bin/env bash
# Compatibility entry point: discovers every registered environment.
# Use --help for selection, duration, output, and viewer options.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python scripts/record_all_envs.py "$@"
