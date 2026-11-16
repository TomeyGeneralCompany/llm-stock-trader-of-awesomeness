#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
uv="$(command -v uv)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"
cat > "$unit_dir/paper-trader.service" <<EOF
[Unit]
Description=Autonomous paper trader and continuous market intelligence
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory="$root"
Environment="PYTHONPATH=$root/backend"
ExecStart="$uv" run --locked python -m app.paper_trader run
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now paper-trader.service

systemctl --user restart paper-trader.service
