"""Kronos prediction service for Sniper_poly_bot.

Runs as a separate process. Every 5 minutes (at the start of a new Polymarket
window) generates a fan of 20 forecast paths for BTC and writes consensus
to kronos_signal.json. The bot reads this file when deciding to fire.

Usage (in separate terminal):
    python kronos_service.py [--device mps|cpu] [--paths 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# Path to local Kronos repo (adjust if needed)
KRONOS_PATH = Path.home() / "Desktop" / "Projects" / "Kronos"
if not KRONOS_PATH.exists():
    print(f"[!] Kronos repo not found at {KRONOS_PATH}")
    print("    Edit KRONOS_PATH at top of kronos_service.py")
    sys.exit(1)
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

WINDOW_SECS = 300
SIGNAL_FILE = Path("kronos_signal.json")
LOOKBACK = 400      # candles fed to model
PRED_LEN = 1        # forecast 1 candle ahead (5 minutes)
TIMEFRAME = "5m"


def next_window_start(now_ts: float) -> int:
    """Return UNIX ts of the next 5-min boundary."""
    return int(now_ts - (now_ts % WINDOW_SECS) + WINDOW_SECS)


def fetch_btc_candles(exchange: ccxt.Exchange, n: int) -> pd.DataFrame:
    bars = exchange.fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, limit=n)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["amount"] = df["close"] * df["volume"]
    return df


def generate_fan(predictor: KronosPredictor, df: pd.DataFrame, n_paths: int) -> np.ndarray:
    """Generate N independent forecast paths. Returns array shape (n_paths, PRED_LEN)."""
    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts = df.iloc[-LOOKBACK:]["ts"].reset_index(drop=True)
    last_ts = x_ts.iloc[-1]
    y_ts = pd.Series(pd.date_range(
        start=last_ts + pd.Timedelta(minutes=5),
        periods=PRED_LEN,
        freq="5min",
    ))

    paths = []
    for _ in range(n_paths):
        p = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1,
            verbose=False,
        )
        paths.append(p["close"].values)
    return np.array(paths)


def build_signal(paths: np.ndarray, last_price: float, window_ts: int) -> dict:
    """Convert fan of paths to a structured signal."""
    final_prices = paths[:, -1]
    votes_up = int((final_prices > last_price).sum())
    n_paths = len(paths)
    votes_down = n_paths - votes_up
    mean_final = float(final_prices.mean())
    mean_pct = (mean_final - last_price) / last_price * 100.0
    p10, p90 = np.percentile(final_prices, [10, 90])
    spread_pct = float((p90 - p10) / last_price * 100.0)

    consensus = max(votes_up, votes_down) / n_paths
    if consensus >= 0.70:
        direction = "UP" if votes_up > votes_down else "DOWN"
    else:
        direction = "NEUTRAL"

    return {
        "window_ts": window_ts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_price": round(last_price, 2),
        "n_paths": n_paths,
        "votes_up": votes_up,
        "votes_down": votes_down,
        "consensus": round(consensus, 3),
        "direction": direction,           # UP / DOWN / NEUTRAL
        "mean_pred_pct": round(mean_pct, 4),
        "spread_pct": round(spread_pct, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    args = ap.parse_args()

    print(f"[Kronos] Loading {args.model} on {args.device}...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512, device=args.device)
    print(f"[Kronos] Ready. Will publish signals to {SIGNAL_FILE.absolute()}")

    exchange = ccxt.binance()

    while True:
        try:
            # Wait until ~10s past the start of the next 5-min window
            # (give the new 5m candle a moment to form on Binance)
            now = time.time()
            next_ws = next_window_start(now)
            wait_until = next_ws + 10  # 10s into the new window
            wait = wait_until - now
            if wait > 0:
                print(f"[Kronos] Sleeping {wait:.0f}s until next window @ "
                      f"{datetime.fromtimestamp(next_ws, timezone.utc).strftime('%H:%M')} UTC")
                time.sleep(wait)

            window_ts = next_window_start(time.time()) - WINDOW_SECS  # the window we just entered

            t0 = time.time()
            print(f"[Kronos] Window {window_ts}: fetching candles...")
            df = fetch_btc_candles(exchange, LOOKBACK + 10)
            last_price = float(df["close"].iloc[-1])

            print(f"[Kronos]   Generating {args.paths} paths...")
            paths = generate_fan(predictor, df, args.paths)

            sig = build_signal(paths, last_price, window_ts)
            sig["inference_secs"] = round(time.time() - t0, 1)

            SIGNAL_FILE.write_text(json.dumps(sig, indent=2))
            print(
                f"[Kronos] ✓ window={window_ts} dir={sig['direction']} "
                f"votes={sig['votes_up']}/{args.paths} "
                f"mean={sig['mean_pred_pct']:+.3f}% "
                f"spread={sig['spread_pct']:.3f}% "
                f"({sig['inference_secs']}s)"
            )

        except KeyboardInterrupt:
            print("\n[Kronos] Stopped by user")
            break
        except Exception as e:
            print(f"[Kronos] Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
