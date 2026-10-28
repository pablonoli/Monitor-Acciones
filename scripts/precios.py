"""Baja solo los precios actuales (un pedido para todas las empresas) y arma data/precios.json."""
import json, math, datetime as dt
import yfinance as yf

ALT = {"BK": "BNY", "MMC": "MRSH", "ELP": "ELPC", "XROX": "XRX", "TRVV": "TRV"}
co = json.load(open("data/datos.json"))["co"]
tickers = [t for t, d in co.items() if d.get("status") == "ok"]
sym = {t: ALT.get(t, t) for t in tickers}
inv = {v: k for k, v in sym.items()}

df = yf.download(list(sym.values()), period="1d", interval="5m", group_by="ticker", progress=False, threads=True, prepost=False)
out = {}
for s, t in inv.items():
    try:
        col = df[s]["Close"].dropna()
        if len(col):
            v = float(col.iloc[-1])
            if math.isfinite(v) and v > 0:
                out[t] = round(v, 4)
    except Exception:
        pass

# los que no salieron en el pedido grande se piden de a uno
for t in [t for t in tickers if t not in out][:40]:
    try:
        v = float(yf.Ticker(sym[t]).fast_info["lastPrice"])
        if math.isfinite(v) and v > 0:
            out[t] = round(v, 4)
    except Exception:
        pass

if len(out) < len(tickers) * 0.5:
    raise SystemExit(f"Solo {len(out)} de {len(tickers)} precios: no se guarda para no pisar los datos")
json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "p": out}, open("data/precios.json", "w"))
print(f"{len(out)} de {len(tickers)} precios")
