"""Cierres semanales de los últimos 5 años (ajustados por splits, no por dividendos) -> data/semanal.json.
Con ellos el sitio dibuja el P/E histórico semanal: precio de cada semana x acciones / ganancia neta de 12 meses (interpolada entre cierres de ejercicio).
Solo empresas que cotizan en dólares: en otra moneda el precio no se puede dividir por una ganancia ya convertida."""
import json, math, datetime as dt
import pandas as pd

ALT = {"BK": "BNY", "MMC": "MRSH", "ELP": "ELPC", "XROX": "XRX", "TRVV": "TRV"}


def main():
    import yfinance as yf
    co = json.load(open("data/datos.json"))["co"]
    tickers = [t for t, d in co.items() if d.get("status") == "ok" and (not d.get("cur") or d.get("cur") == "USD")]
    sym = {t: ALT.get(t, t) for t in tickers}
    inv = {v: k for k, v in sym.items()}
    try:
        previo = json.load(open("data/semanal.json"))
    except Exception:
        previo = {}
    ser = {}
    syms = list(sym.values())
    for i in range(0, len(syms), 80):
        lote = syms[i:i + 80]
        try:
            df = yf.download(lote, period="5y", interval="1wk", group_by="ticker", progress=False, threads=True, auto_adjust=False)
        except Exception as e:
            print("lote falló:", e)
            continue
        for s in lote:
            try:
                x = (df[s] if len(lote) > 1 else df)["Close"].dropna()
                x = x[x > 0]
                if len(x) >= 52:
                    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
                    ser[inv[s]] = x
            except Exception:
                pass
    if len(ser) < len(tickers) * 0.5:
        raise SystemExit(f"Solo {len(ser)} de {len(tickers)}: no se guarda para no pisar los datos")
    idx = sorted(set().union(*[set(x.index) for x in ser.values()]))
    out = {}
    for t, x in ser.items():
        x = x[~x.index.duplicated(keep="last")].reindex(idx)
        out[t] = [None if not math.isfinite(v) else round(float(v), 2) for v in x.values]
    # los que no se pudieron bajar hoy no se muestran con una fecha vieja: se vuelven a intentar la próxima vez
    json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "d": [d.date().isoformat() for d in idx], "t": out}, open("data/semanal.json", "w"), separators=(",", ":"))
    print(f"{len(out)} de {len(tickers)} con cierres semanales")


if __name__ == "__main__":
    main()
