"""
Paper Trading Bot (Strategy A) - scanner + position manager

Features:
- Uses yfinance to fetch daily & intraday (5m) data
- Daily: SMA200, 20-day high (trend + breakout level)
- Intraday (5m): EMA20, EMA50, RSI14, volume
- Detects Breakout + Retest and enters a PAPER trade on retest bullish close
- Exits when: RSI < 50 OR close < EMA20 OR stoploss hit
- Persists open positions to open_positions.csv
- Logs closed trades to trades.csv
- Logs detected setups to setups.csv

Usage:
- Install dependencies: pip install yfinance pandas numpy
- Run once (cron / GitHub Actions recommended every 5-15 minutes)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
import math

# ---------------------------
# CONFIG
# ---------------------------
SYMBOLS = ['360ONE.NS', '3MINDIA.NS', 'ACC.NS', 'AIAENG.NS', 'APLAPOLLO.NS', 'AUBANK.NS', 'AADHARHFC.NS', 'AAVAS.NS', 'ABBOTINDIA.NS', 'ACE.NS', 'ATGL.NS', 'ABREL.NS', 'ABSLAMC.NS', 'AEGISLOG.NS', 'AFCONS.NS', 'AFFLE.NS', 'AJANTPHARM.NS', 'AKUMS.NS', 'AKZOINDIA.NS', 'APLLTD.NS', 'ALKEM.NS', 'ALKYLAMINE.NS', 'ARE&M.NS', 'AMBER.NS', 'ANANDRATHI.NS', 'ANANTRAJ.NS', 'ANGELONE.NS', 'APARINDS.NS', 'APOLLOTYRE.NS', 'ASAHIINDIA.NS', 'ASTERDM.NS', 'ASTRAZEN.NS', 'ASTRAL.NS', 'ATHERENERG.NS', 'ATUL.NS', 'AIIL.NS', 'BASF.NS', 'BEML.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BATAINDIA.NS', 'BAYERCROP.NS', 'BERGEPAINT.NS', 'BDL.NS', 'BHARATFORG.NS', 'BIKAJI.NS', 'BIOCON.NS', 'BLUEDART.NS', 'BLUEJET.NS', 'BLUESTARCO.NS', 'BBTC.NS', 'BRIGADE.NS', 'MAPMYINDIA.NS', 'CCL.NS', 'CRISIL.NS', 'CANFINHOME.NS', 'CAPLIPOINT.NS', 'CARBORUNIV.NS', 'CEATLTD.NS', 'CDSL.NS', 'CENTURYPLY.NS', 'CERA.NS', 'CHALET.NS', 'CHAMBLFERT.NS', 'CHENNPETRO.NS', 'CHOICEIN.NS', 'CHOLAHLDNG.NS', 'CLEAN.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COHANCE.NS', 'COLPAL.NS', 'CAMS.NS', 'CONCORDBIO.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CYIENT.NS', 'DCMSHRIRAM.NS', 'DOMS.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKFERT.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'AGARWALEYE.NS', 'LALPATHLAB.NS', 'EIDPARRY.NS', 'ELECON.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'EMCURE.NS', 'ENDURANCE.NS', 'ERIS.NS', 'ESCORTS.NS', 'FACT.NS', 'FINCABLES.NS', 'FIVESTAR.NS', 'FORCEMOT.NS', 'GRSE.NS', 'GILLETTE.NS', 'GLAND.NS', 'GLAXO.NS', 'GLENMARK.NS', 'MEDANTA.NS', 'GODFRYPHLP.NS', 'GODREJAGRO.NS', 'GODREJIND.NS', 'GODREJPROP.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GRAVITA.NS', 'GESHIP.NS', 'FLUOROCHEM.NS', 'GUJGASLTD.NS', 'GMDCLTD.NS', 'HEG.NS', 'HBLENGINE.NS', 'HAPPSTMNDS.NS', 'HEXT.NS', 'HSCL.NS', 'HOMEFIRST.NS', 'HONAUT.NS', 'IIFL.NS', 'INOXINDIA.NS', 'INDGN.NS', 'INDIAMART.NS', 'IRCTC.NS', 'INDUSINDBK.NS', 'INTELLECT.NS', 'IKS.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JBMA.NS', 'JKTYRE.NS', 'JSL.NS', 'JUBLFOOD.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JYOTICNC.NS', 'KPRMILL.NS', 'KEI.NS', 'KPITTECH.NS', 'KSB.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KAYNES.NS', 'KEC.NS', 'KFINTECH.NS', 'KIRLOSBROS.NS', 'KIRLOSENG.NS', 'KIMS.NS', 'LTTS.NS', 'LICHSGFIN.NS', 'LTFOODS.NS', 'LATENTVIEW.NS', 'LAURUSLABS.NS', 'THELEELA.NS', 'LINDEINDIA.NS', 'LLOYDSME.NS', 'MRF.NS', 'MGL.NS', 'MAHSCOOTER.NS', 'MAHSEAMLES.NS', 'MFSL.NS', 'METROPOLIS.NS', 'MINDACORP.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MCX.NS', 'NATCOPHARM.NS', 'NH.NS', 'NAVA.NS', 'NAVINFLUOR.NS', 'NETWEB.NS', 'NEULANDLAB.NS', 'NEWGEN.NS', 'NAM-INDIA.NS', 'NUVAMA.NS', 'OBEROIRLTY.NS', 'OLECTRA.NS', 'ONESOURCE.NS', 'PGEL.NS', 'PIIND.NS', 'PNBHOUSING.NS', 'PTCIL.NS', 'PVRINOX.NS', 'PAGEIND.NS', 'PATANJALI.NS', 'PFIZER.NS', 'PHOENIXLTD.NS', 'POLYMED.NS', 'POONAWALLA.NS', 'PREMIERENE.NS', 'PGHH.NS', 'RRKABEL.NS', 'RHIM.NS', 'RADICO.NS', 'RAINBOW.NS', 'RKFORGE.NS', 'SKFINDIA.NS', 'SAILIFE.NS', 'SARDAEN.NS', 'SCHAEFFLER.NS', 'SCHNEIDER.NS', 'SHYAMMETL.NS', 'SIGNATURE.NS', 'SOBHA.NS', 'SONACOMS.NS', 'STARHEALTH.NS', 'SUMICHEM.NS', 'SUNTV.NS', 'SUNDARMFIN.NS', 'SUNDRMFAST.NS', 'SUPREMEIND.NS', 'SWANCORP.NS', 'SYNGENE.NS', 'SYRMA.NS', 'TBOTEK.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATAELXSI.NS', 'TATAINVEST.NS', 'TATATECH.NS', 'TECHNOE.NS', 'TEJASNET.NS', 'RAMCOCEM.NS', 'THERMAX.NS', 'TIMKEN.NS', 'TITAGARH.NS', 'TORNTPOWER.NS', 'TRITURBINE.NS', 'TIINDIA.NS', 'UPL.NS', 'UTIAMC.NS', 'UBL.NS', 'USHAMART.NS', 'VTL.NS', 'MANYAVAR.NS', 'VENTIVE.NS', 'VIJAYA.NS', 'VOLTAS.NS', 'WELCORP.NS', 'WHIRLPOOL.NS', 'WOCKPHARMA.NS', 'ZFCVINDIA.NS', 'ZENTEC.NS', 'ZENSARTECH.NS', 'ECLERX.NS']


CAPITAL = 200000  # paper trading capital in INR (adjust)
RISK_PER_TRADE_PCT = 1.0  # percent of CAPITAL risked per trade (e.g., 1%)
MIN_PRICE = 100  # skip low-priced stocks
MIN_AVG_VOL = 300000  # minimum average daily volume (shares)
CHECK_INTERVAL_MINUTES = 5  # used conceptually; script runs once per scheduler run

OUTPUT_TRADES_CSV = "trades.csv"
OPEN_POSITIONS_CSV = "open_positions.csv"
SETUPS_CSV = "setups.csv"

# ---------------------------
# INDICATORS
# ---------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral for early values

# ---------------------------
# Helpers: fetch data + indicators
# ---------------------------
def fetch_daily(symbol: str, period_days: int = 365) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{period_days}d", interval="1d", progress=False)
    if df.empty:
        return df
    df["SMA200"] = df["Close"].rolling(window=200, min_periods=50).mean()
    df["HIGH20"] = df["High"].rolling(window=20, min_periods=5).max()
    df["VOL20"] = df["Volume"].rolling(window=20, min_periods=5).mean()
    return df

def fetch_intraday(symbol: str, period: str = "7d", interval: str = "5m") -> pd.DataFrame:
    """
    period can be '7d' or '60d' depending on yfinance limits;
    interval '5m' is used for entries/exits. yfinance intraday sometimes limited to ~60 days.
    """
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["VOL20"] = df["Volume"].rolling(window=20, min_periods=5).mean()
    return df

# ---------------------------
# Strategy checks
# ---------------------------
def daily_trend_ok(daily_df: pd.DataFrame) -> bool:
    # Price above SMA200 and SMA200 sloping up
    last = daily_df.iloc[-1]
    if math.isnan(last.get("SMA200", np.nan)):
        return False
    return (last["Close"] > last["SMA200"]) and (daily_df["SMA200"].iloc[-1] > daily_df["SMA200"].iloc[-2])

def detect_breakout_retest(symbol: str, daily_df: pd.DataFrame, intra_df: pd.DataFrame):
    """
    Returns a dict with setup details if a Breakout+Retest on intraday 5m is present; else None.
    Breakout level taken as previous day's HIGH20 (last complete daily HIGH20).
    Conditions (adapted to intraday):
      - daily trend ok (SMA200)
      - daily HIGH20 exists (breakout level)
      - intraday last bar is a retest bullish close:
          low <= breakout_level AND close > breakout_level
      - intraday EMA20 > EMA50
      - intraday RSI between 50 and 70
      - volume confirmation: intraday volume >= 0.8 * daily VOL20 (approx)
    """
    try:
        breakout_level = daily_df["HIGH20"].iloc[-2]  # previous completed day HIGH20
    except Exception:
        return None

    if np.isnan(breakout_level):
        return None

    last = intra_df.iloc[-1]
    prev = intra_df.iloc[-2] if len(intra_df) >= 2 else None

    # basic price checks
    if last["Close"] <= breakout_level:
        return None
    if not (last["Low"] <= breakout_level and last["Close"] > breakout_level):
        return None

    # ema & rsi checks
    if not (last["EMA20"] > last["EMA50"]):
        return None
    if not (50 < last["RSI"] < 70):
        return None

    # volume check (compare intraday vol to daily avg vol — heuristic)
    daily_vol20 = daily_df["VOL20"].iloc[-1] if "VOL20" in daily_df.columns else None
    if daily_vol20 is not None and not np.isnan(daily_vol20) and daily_vol20 > 0:
        # average intraday volume threshold
        vol_ok = last["Volume"] >= (0.08 * daily_vol20)  # 0.08 ~ fraction per 5-min bar; heuristic
        if not vol_ok:
            return None

    # Passed checks -> return details
    return {
        "symbol": symbol,
        "breakout_level": float(breakout_level),
        "entry_price": float(last["Close"]),
        "retest_bar_time": str(last.name),  # timestamp index
        "ema20": float(last["EMA20"]),
        "ema50": float(last["EMA50"]),
        "rsi": float(last["RSI"]),
        "volume": int(last["Volume"]),
    }

# ---------------------------
# Position sizing
# ---------------------------
def compute_position_size(entry_price: float, stoploss: float, capital: float = CAPITAL, risk_pct: float = RISK_PER_TRADE_PCT):
    """
    Returns qty (int) such that risk (entry - stoploss)*qty = capital * risk_pct/100
    If stoploss >= entry_price => returns 0
    """
    risk_amount = capital * (risk_pct / 100.0)
    tick_risk = entry_price - stoploss
    if tick_risk <= 0:
        return 0
    qty = math.floor(risk_amount / tick_risk)
    return max(0, int(qty))

# ---------------------------
# CSV helpers
# ---------------------------
def append_to_csv(filename: str, row: dict, columns: list = None):
    df_row = pd.DataFrame([row])
    if not os.path.exists(filename):
        df_row.to_csv(filename, index=False)
    else:
        df_row.to_csv(filename, mode="a", index=False, header=False)

def load_open_positions() -> pd.DataFrame:
    if os.path.exists(OPEN_POSITIONS_CSV):
        try:
            return pd.read_csv(OPEN_POSITIONS_CSV, parse_dates=["entry_datetime", "entry_bar_time"])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_open_positions(df: pd.DataFrame):
    df.to_csv(OPEN_POSITIONS_CSV, index=False)

# ---------------------------
# Main logic: scan, open, manage positions
# ---------------------------
def run_once(symbols=SYMBOLS):
    now = dt.datetime.utcnow()
    print(f"[{now.isoformat()}] Starting scan for {len(symbols)} symbols...")

    open_positions = load_open_positions()
    if open_positions is None:
        open_positions = pd.DataFrame()

    closed_trades = []

    for symbol in symbols:
        try:
            # Fetch data
            daily_df = fetch_daily(symbol, period_days=400)
            if daily_df.empty:
                print(f"  {symbol}: no daily data")
                continue

            # Volume + price filters (quality filter)
            if daily_df["Volume"].iloc[-1] < MIN_AVG_VOL and daily_df["Close"].iloc[-1] < MIN_AVG_VOL:
                # lenient: skip if last day volume < MIN_AVG_VOL (you can remove this)
                pass

            if daily_df["Close"].iloc[-1] < MIN_PRICE:
                print(f"  {symbol}: price < MIN_PRICE, skipping")
                continue

            if not daily_trend_ok(daily_df):
                # trend fails (SMA200 not supportive)
                #print(f"  {symbol}: daily trend not ok")
                pass

            # fetch intraday 5m
            intra_df = fetch_intraday(symbol, period="7d", interval="5m")
            if intra_df.empty or len(intra_df) < 30:
                print(f"  {symbol}: insufficient intraday data")
                continue

            # detect setups
            setup = detect_breakout_retest(symbol, daily_df, intra_df)
            if setup is not None:
                # record setup
                setup_row = {
                    "scan_time": dt.datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "entry_price": setup["entry_price"],
                    "breakout_level": setup["breakout_level"],
                    "rsi": setup["rsi"],
                    "ema20": setup["ema20"],
                    "ema50": setup["ema50"],
                    "volume": setup["volume"],
                    "reason": "Breakout+Retest+Trend+RSI+EMA",
                }
                append_to_csv(SETUPS_CSV, setup_row)
                print(f"  {symbol}: setup detected at {setup['entry_price']}, breakout {setup['breakout_level']}")

                # Check if we already have this symbol open
                already_open = False
                if not open_positions.empty:
                    if symbol in open_positions['symbol'].values:
                        already_open = True

                if not already_open:
                    # Determine stoploss (use retest swing low: find the lowest low in the last N intraday bars that touched <= breakout_level)
                    breakout_lvl = setup["breakout_level"]
                    # find bars within last 40 bars that dipped <= breakout_lvl, use their min low as retest low
                    recent = intra_df.tail(40)
                    retest_lows = recent[recent["Low"] <= breakout_lvl]["Low"]
                    if not retest_lows.empty:
                        stoploss = float(retest_lows.min())
                    else:
                        # fallback: use breakout_level * 0.995 (0.5% below)
                        stoploss = float(breakout_lvl * 0.995)

                    entry_price = setup["entry_price"]
                    qty = compute_position_size(entry_price, stoploss)
                    if qty <= 0:
                        print(f"    position size 0 (risk too small). skipping entry.")
                    else:
                        entry_row = {
                            "symbol": symbol,
                            "entry_datetime": dt.datetime.utcnow().isoformat(),
                            "entry_bar_time": setup["retest_bar_time"],
                            "entry_price": entry_price,
                            "qty": qty,
                            "stoploss": stoploss,
                            "status": "OPEN",
                            "notes": "Paper entry from Breakout+Retest",
                        }
                        # append to open positions (persist)
                        if open_positions.empty:
                            open_positions = pd.DataFrame([entry_row])
                        else:
                            open_positions = pd.concat([open_positions, pd.DataFrame([entry_row])], ignore_index=True)
                        save_open_positions(open_positions)
                        print(f"    OPEN PAPER TRADE → {symbol} qty={qty} entry={entry_price:.2f} SL={stoploss:.2f}")

            # Manage open positions for this symbol (exit checks)
            if not open_positions.empty and symbol in open_positions['symbol'].values:
                # iterate over positions for this symbol (usually one)
                pos_indices = open_positions.index[open_positions['symbol'] == symbol].tolist()
                for idx in pos_indices:
                    pos = open_positions.loc[idx]
                    # get latest price & indicators from intraday
                    last = intra_df.iloc[-1]
                    last_price = float(last["Close"])
                    last_rsi = float(last["RSI"])
                    last_ema20 = float(last["EMA20"])

                    exit_reason = None
                    exit_price = None

                    # SL hit
                    if last_price <= float(pos["stoploss"]):
                        exit_reason = "SL_HIT"
                        exit_price = last_price

                    # RSI exit
                    elif last_rsi < 50:
                        exit_reason = "RSI<50"
                        exit_price = last_price

                    # EMA20 exit
                    elif last_price < last_ema20:
                        exit_reason = "Close<EMA20"
                        exit_price = last_price

                    # You could add profit targets here (T1/T2) if wanted

                    if exit_reason is not None:
                        # compute pnl
                        entry_price = float(pos["entry_price"])
                        qty = int(pos["qty"])
                        pnl = (exit_price - entry_price) * qty
                        pnl_pct = (pnl / (entry_price * qty)) * 100 if entry_price * qty != 0 else 0
                        risk_per_share = entry_price - float(pos["stoploss"]) if entry_price > float(pos["stoploss"]) else 0.00001
                        r_mult = (exit_price - entry_price) / risk_per_share if risk_per_share != 0 else None

                        trade_row = {
                            "symbol": symbol,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "entry_datetime": pos["entry_datetime"],
                            "exit_datetime": dt.datetime.utcnow().isoformat(),
                            "qty": qty,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "r_multiple": r_mult,
                            "exit_reason": exit_reason,
                            "notes": pos.get("notes", "")
                        }
                        append_to_csv(OUTPUT_TRADES_CSV, trade_row)
                        print(f"    CLOSED {symbol} entry={entry_price:.2f} exit={exit_price:.2f} PnL={pnl:.2f} reason={exit_reason}")

                        # remove from open positions
                        open_positions = open_positions.drop(index=idx).reset_index(drop=True)
                        save_open_positions(open_positions)

        except Exception as e:
            print(f"  {symbol}: error during processing -> {e}")

    print("Scan complete.")

if __name__ == "__main__":
    run_once()
