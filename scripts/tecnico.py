"""Baja 2 años de precios diarios de Yahoo Finance y calcula los indicadores técnicos -> data/tecnico.json.
Se usa el precio en dólares del título original (ADR o acción), ajustado por dividendos y splits."""
import json, math, os, datetime as dt
import numpy as np
import pandas as pd

ALT = {"BK": "BNY", "MMC": "MRSH", "ELP": "ELPC", "XROX": "XRX", "TRVV": "TRV"}
SEMANAS = 104


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    if dn.iloc[-1] == 0:
        return 100.0 if up.iloc[-1] > 0 else 50.0
    return float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1]))


def r(v, n=4):
    try:
        v = float(v)
    except Exception:
        return None
    return round(v, n) if math.isfinite(v) else None


def calc(close, vol=None):
    """close: pandas Series con fechas; devuelve el diccionario de indicadores o None si hay muy poco historial."""
    c = close.dropna()
    c = c[c > 0]
    if len(c) < 60:
        return None
    last = float(c.iloc[-1])
    s50 = c.rolling(50).mean()
    s200 = c.rolling(200).mean()
    m50 = float(s50.iloc[-1])
    m200 = float(s200.iloc[-1]) if len(c) >= 200 and math.isfinite(s200.iloc[-1]) else None
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    y = c.iloc[-252:]
    hi, lo = float(y.max()), float(y.min())
    def ret(n):
        return r(last / float(c.iloc[-1 - n]) - 1) if len(c) > n else None
    # cruce de las medias de 50 y 200 ruedas en los últimos 60 días
    cruce = None
    if m200 is not None:
        dif = (s50 - s200).dropna().iloc[-61:]
        sg = np.sign(dif.values)
        for i in range(len(sg) - 1, 0, -1):
            if sg[i] != sg[i - 1] and sg[i] != 0:
                cruce = {"t": "dorada" if sg[i] > 0 else "muerte", "d": len(sg) - 1 - i}
                break
    if m200 is not None:
        tr = "up" if last > m50 > m200 else "down" if last < m50 < m200 else "mix"
    else:
        tr = "up" if last > m50 else "down"
    rv = None
    if vol is not None:
        v = vol.dropna()
        if len(v) >= 55 and v.iloc[-50:].mean() > 0:
            rv = r(v.iloc[-5:].mean() / v.iloc[-50:].mean(), 2)
    w = c.resample("W-FRI").last().dropna().iloc[-SEMANAS:]
    return {
        "c": r(last, 2), "tr": tr, "s50": r(m50, 2), "s200": r(m200, 2) if m200 else None,
        "d50": r(last / m50 - 1), "d200": r(last / m200 - 1) if m200 else None,
        "rsi": r(rsi(c), 1), "macd": r(macd.iloc[-1], 3), "sig": r(sig.iloc[-1], 3), "mh": r(macd.iloc[-1] - sig.iloc[-1], 3),
        "hi": r(hi, 2), "lo": r(lo, 2), "dhi": r(last / hi - 1), "dlo": r(last / lo - 1),
        "r1": ret(21), "r3": ret(63), "r6": ret(126), "r12": ret(252),
        "cruce": cruce, "rv": rv,
        "wend": c.index[-1].date().isoformat(), "w": [r(x, 2) for x in w.values],
    }


def main():
    import yfinance as yf
    co = json.load(open("data/datos.json"))["co"]
    tickers = [t for t, d in co.items() if d.get("status") == "ok"]
    sym = {t: ALT.get(t, t) for t in tickers}
    inv = {v: k for k, v in sym.items()}
    try:
        previo = json.load(open("data/tecnico.json"))["t"]
    except Exception:
        previo = {}
    out = {}
    syms = list(sym.values())
    for i in range(0, len(syms), 80):
        lote = syms[i:i + 80]
        try:
            df = yf.download(lote, period="2y", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=True)
        except Exception as e:
            print("lote falló:", e)
            continue
        for s in lote:
            try:
                x = df[s] if len(lote) > 1 else df
                res = calc(x["Close"], x["Volume"] if "Volume" in x else None)
                if res:
                    out[inv[s]] = res
            except Exception:
                pass
    faltan = [t for t in tickers if t not in out]
    for t in faltan[:40]:   # los que no salieron en el pedido grande se piden de a uno
        try:
            x = yf.Ticker(sym[t]).history(period="2y", interval="1d", auto_adjust=True)
            res = calc(x["Close"].tz_localize(None) if x.index.tz is not None else x["Close"], x.get("Volume"))
            if res:
                out[t] = res
        except Exception:
            pass
    if len(out) < len(tickers) * 0.5:
        raise SystemExit(f"Solo {len(out)} de {len(tickers)}: no se guarda para no pisar los datos")
    for t, v in previo.items():   # lo que no se pudo bajar hoy conserva el dato anterior
        out.setdefault(t, v)
    json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "t": out}, open("data/tecnico.json", "w"), separators=(",", ":"))
    print(f"{len(out)} de {len(tickers)} con indicadores")


if __name__ == "__main__":
    main()
