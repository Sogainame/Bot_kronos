"""Single-command launcher for Bot_kronos.

Spawns kronos_service.py and bot.py as subprocesses in one terminal.
Logs from both are streamed live with coloured prefixes.
Ctrl+C kills both cleanly.

Usage:
    python3 run.py                            # DRY mode, Kronos ON, asset=btc
    python3 run.py --live                     # LIVE trading
    python3 run.py --asset btc,eth            # multiple assets
    python3 run.py --mode aggressive          # change Kelly mode
    python3 run.py --no-kronos                # disable Kronos filter
    python3 run.py --device cpu               # if MPS not available
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ANSI colours
RESET = "\033[0m"
KRONOS_COLOR = "\033[36m"  # cyan
BOT_COLOR = "\033[33m"     # yellow
ERR_COLOR = "\033[31m"     # red
INFO_COLOR = "\033[32m"    # green


def stream_output(proc: subprocess.Popen, prefix: str, color: str) -> None:
    """Read subprocess output line-by-line and print with coloured prefix."""
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:
            line = str(raw)
        if not line:
            continue
        print(f"{color}[{prefix}]{RESET} {line}", flush=True)
    proc.stdout.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Bot_kronos launcher")
    ap.add_argument("--asset", default="btc", help="btc | btc15m | eth | sol | xrp | doge | all | comma-list")
    ap.add_argument("--mode", default="safe", choices=["safe", "aggressive", "degen"])
    ap.add_argument("--max-bet", type=float, default=50.0)
    ap.add_argument("--live", action="store_true", help="LIVE trading (default: DRY)")
    ap.add_argument("--no-kronos", action="store_true", help="Disable Kronos filter")
    ap.add_argument("--kronos-driven", action="store_true",
                    help="Kronos-driven mode: bot trades direction from Kronos, no other filters")
    ap.add_argument("--timeframe", default="5", choices=["5", "15"],
                    help="Window length in minutes: 5 or 15. Auto-selects asset and kronos window.")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--paths", type=int, default=20, help="Forecast paths in fan")
    args = ap.parse_args()

    here = Path(__file__).parent.resolve()

    # Resolve asset and kronos window based on --timeframe
    asset_arg = args.asset
    if args.timeframe == "15" and asset_arg == "btc":
        asset_arg = "btc15m"  # auto-switch to 15-min BTC config

    # Build commands
    py = sys.executable
    kronos_cmd = [py, "-u", str(here / "kronos_service.py"),
                  "--device", args.device,
                  "--paths", str(args.paths),
                  "--window-minutes", args.timeframe]

    bot_cmd = [py, "-u", str(here / "bot.py"),
               "--asset", asset_arg,
               "--mode", args.mode,
               "--max-bet", str(args.max_bet)]
    if args.live:
        bot_cmd.append("--live")
    if args.no_kronos:
        bot_cmd.append("--no-kronos")
    if args.kronos_driven:
        bot_cmd.append("--kronos-driven")

    print(f"{INFO_COLOR}{'=' * 64}{RESET}")
    print(f"{INFO_COLOR}  Bot_kronos launcher{RESET}")
    kronos_mode = ("KRONOS-DRIVEN" if args.kronos_driven else
                   "OFF" if args.no_kronos else "ON (filter)")
    print(f"{INFO_COLOR}  Mode: {'LIVE' if args.live else 'DRY'} | "
          f"Kronos: {kronos_mode} | "
          f"Asset: {asset_arg} | Timeframe: {args.timeframe}m | Device: {args.device}{RESET}")
    print(f"{INFO_COLOR}{'=' * 64}{RESET}\n")

    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    def cleanup(*_: object) -> None:
        print(f"\n{INFO_COLOR}[launcher] Stopping all processes...{RESET}")
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print(f"{INFO_COLOR}[launcher] Done.{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start Kronos first
    if not args.no_kronos:
        print(f"{INFO_COLOR}[launcher] Starting Kronos service...{RESET}")
        kp = subprocess.Popen(kronos_cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1)
        procs.append(kp)
        t = threading.Thread(target=stream_output, args=(kp, "KRONOS", KRONOS_COLOR),
                             daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)  # give Kronos a head start to load the model
    else:
        print(f"{INFO_COLOR}[launcher] Kronos disabled by --no-kronos{RESET}")

    # Start bot
    print(f"{INFO_COLOR}[launcher] Starting Sniper bot...{RESET}")
    bp = subprocess.Popen(bot_cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, bufsize=1)
    procs.append(bp)
    t = threading.Thread(target=stream_output, args=(bp, "BOT", BOT_COLOR),
                         daemon=True)
    t.start()
    threads.append(t)

    # Wait — if any subprocess dies, kill the rest
    try:
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"{ERR_COLOR}[launcher] One process exited "
                          f"(code={p.returncode}). Stopping...{RESET}")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
