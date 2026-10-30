"""Cierres diarios de los últimos 5 años (ajustados por splits, no por dividendos) -> data/diario.json.
Con ellos el sitio dibuja el P/E histórico diario: precio de cada rueda x acciones / ganancia neta de 12 meses (interpolada entre cierres de ejercicio).
El archivo pesa unos 2 MB, así que se rehace solo si tiene más de 6 días (o con --forzar).
Empresas que reportan en otra moneda (euro, yen...): el precio cotiza en dólares pero la ganancia está en su moneda, así que se guarda también el tipo de cambio diario
(unidades de esa moneda por US$) y el sitio convierte la ganancia con el cambio de cada fecha. El peso argentino queda afuera: con la inflación esa conversión no tiene sentido."""
import json, math, sys, datetime as dt
import pandas as pd

ALT = {"BK": "BNY", "MMC": "MRSH", "ELP": "ELPC", "XROX": "XRX", "TRVV": "TRV"}


def main():
    import yfinance as yf
    try:
        prev = json.load(open("data/diario.json"))
        edad = dt.datetime.utcnow() - dt.datetime.fromisoformat(prev["updated"].rstrip("Z"))
        if edad < dt.timedelta(days=6) and "fx" in prev and "--forzar" not in sys.argv:   # sin "fx" es un archivo de formato viejo: se rehace
            print("diario.json es reciente: no se rehace")
            return
    except Exception:
        pass
    co = json.load(open("data/datos.json"))["co"]
    tickers = [t for t, d in co.items() if d.get("status") == "ok" and d.get("cur") != "ARS"]
    sym = {t: ALT.get(t, t) for t in tickers}
    inv = {v: k for k, v in sym.items()}
    ser = {}
    syms = list(sym.values())
    for i in range(0, len(syms), 80):
        lote = syms[i:i + 80]
        try:
            df = yf.download(lote, period="5y", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
        except Exception as e:
            print("lote falló:", e)
            continue
        try:   # el pedido de 5 años a veces llega sin la última rueda: se completa con uno corto
            df = df.combine_first(yf.download(lote, period="5d", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False))
        except Exception as e:
            print("completar últimos días falló:", e)
        for s in lote:
            try:
                x = (df[s] if len(lote) > 1 else df)["Close"].dropna()
                x = x[x > 0]
                if len(x) >= 250:
                    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
                    ser[inv[s]] = x
            except Exception:
                pass
    # tipo de cambio diario de cada moneda en que reportan (ej. EUR=X = euros por 1 US$); sin cambio no se muestra esa empresa
    fxs = {}
    mon = sorted({co[t].get("cur") for t in ser if co[t].get("cur") not in (None, "USD")})
    for i in range(0, len(mon), 40):
        lote = [m + "=X" for m in mon[i:i + 40]]
        try:
            df = yf.download(lote, period="5y", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
            for m in mon[i:i + 40]:
                try:
                    x = (df[m + "=X"] if len(lote) > 1 else df)["Close"].dropna()
                    x = x[x > 0]
                    if len(x) >= 250:
                        x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
                        fxs[m] = x[~x.index.duplicated(keep="last")]
                except Exception:
                    pass
        except Exception as e:
            print("cambio falló:", e)
    for t in [t for t in ser if co[t].get("cur") not in (None, "USD") and co[t].get("cur") not in fxs]:
        del ser[t]
    if len(ser) < len(tickers) * 0.5:
        raise SystemExit(f"Solo {len(ser)} de {len(tickers)}: no se guarda para no pisar los datos")
    idx = sorted(set().union(*[set(x.index) for x in ser.values()]))
    out = {}
    for t, x in ser.items():
        x = x[~x.index.duplicated(keep="last")].reindex(idx)
        out[t] = [None if not math.isfinite(v) else round(float(v), 2) for v in x.values]
    # los que no se pudieron bajar hoy no se muestran con una fecha vieja: se vuelven a intentar la próxima vez
    fxo = {}
    for m, x in fxs.items():
        x = x.reindex(idx).ffill().bfill()
        fxo[m] = [round(float(v), 5) if math.isfinite(v) else None for v in x.values]
    json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "d": [d.date().isoformat() for d in idx], "t": out, "fx": fxo}, open("data/diario.json", "w"), separators=(",", ":"))
    print(f"{len(out)} de {len(tickers)} con cierres diarios; cambio de {len(fxo)} monedas")


if __name__ == "__main__":
    main()
