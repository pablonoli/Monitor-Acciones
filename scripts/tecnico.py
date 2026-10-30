"""Baja 5 años de precios diarios (hacen falta ~4 para la media de 200 semanas) de Yahoo Finance y calcula los indicadores técnicos -> data/tecnico.json.
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


def soporte(c, ma, low, look, slope_n, near_hi, min_ext, toques=False):
    """¿La media funciona como soporte? Mira los últimos `look` períodos.
    Tendencia alcista = la media sube (>= 2% en `slope_n` períodos), el precio estuvo sobre ella en 75% o más del tiempo, se alejó al menos `min_ext` por encima
    y la cruzó 2 veces o menos. Si además hoy está entre -2% y +near_hi de la media -> "sop" (apoyada en la media: la está testeando desde arriba).
    "perd": venía en esa tendencia y hoy cerró más de 2% por debajo. "lat": cruza la media seguido (3 o más cruces) y hoy está cerca: no sirve como soporte."""
    s = pd.concat([c, ma], axis=1, keys=["c", "m"]).dropna()
    if len(s) < max(slope_n + 5, int(look * 0.8)):
        return None
    s = s.iloc[-look:]
    rel = (s["c"] / s["m"] - 1).values
    mm = s["m"].values
    pend = mm[-1] / mm[max(0, len(mm) - 1 - slope_n)] - 1
    cruces = int((np.diff(np.sign(rel)) != 0).sum())
    d = float(rel[-1])
    base = {"p": r(pend), "n": cruces, "x": r(d)}
    def tendencia(rr):
        return len(rr) > 10 and pend >= 0.02 and (rr > 0).mean() >= 0.75 and rr.max() >= min_ext and int((np.diff(np.sign(rr)) != 0).sum()) <= 2
    if tendencia(rel) and -0.02 <= d <= near_hi:
        base["e"] = "sop"
        if toques and low is not None:
            l = low.reindex(s.index).ffill().values
            m = mm
            hit = (l <= m * 1.02) & (rel >= -0.01)
            n, last = 0, -99
            for i in np.where(hit[:-1])[0]:
                if i - last > 5:
                    n += 1
                last = i
            base["t"] = n
        return base
    if len(rel) > 12 and pend >= 0 and d < -0.02 and tendencia(rel[:-10]):
        base["e"] = "perd"
        return base
    if cruces >= 3 and abs(d) <= near_hi:
        base["e"] = "lat"
        return base
    return None


def calc(close, vol=None, high=None, low=None):
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
    # máximos y mínimos con los extremos de cada rueda (High/Low); si no vienen, con los cierres
    hh = high.dropna() if high is not None else c
    ll = low.dropna() if low is not None else c
    hh, ll = hh[hh > 0], ll[ll > 0]
    hi, lo = float(max(hh.iloc[-252:].max(), last)), float(min(ll.iloc[-252:].min(), last))
    # máximos anteriores (picos): máximos semanales que fueron el más alto de las 8 semanas anteriores y posteriores.
    # Se quedan los que están por encima del precio y se agrupan los que están a menos de 1,5% entre sí (queda el más bajo, el primero que se encuentra al subir).
    hw = hh.resample("W-FRI").max().dropna()
    v = hw.values
    picos = []
    for i in range(len(v)):
        if i <= len(v) - 3 and v[i] >= v[max(0, i - 8):i + 9].max() and v[i] > last * 1.001:   # se ignoran las 2 últimas semanas: son el precio de hoy, no un máximo anterior
            picos.append((float(v[i]), hw.index[i].strftime("%Y-%m")))
    picos.sort()
    sw = []
    for val, ym in picos:
        if not sw or val > sw[-1][0] * 1.015:
            sw.append([r(val, 2), ym])
    sw = sw[:4]
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
    wk = c.resample("W-FRI").last().dropna()   # cierres semanales (la última barra puede ser una semana en curso)
    s10w = float(wk.iloc[-10:].mean()) if len(wk) >= 10 else None
    s200w = float(wk.iloc[-200:].mean()) if len(wk) >= 200 else None
    w = wk.iloc[-SEMANAS:]
    so = sow = None
    if m200 is not None:
        so = soporte(c, s200, ll if low is not None else None, 120, 60, 0.03, 0.08, True)
    if len(wk) >= 230:
        sow = soporte(wk, wk.rolling(200).mean(), None, 52, 26, 0.05, 0.10)
    return {
        "c": r(last, 2), "tr": tr, "s50": r(m50, 2), "s200": r(m200, 2) if m200 else None,
        "s10w": r(s10w, 2), "s200w": r(s200w, 2) if s200w else None,
        "d50": r(last / m50 - 1), "d200": r(last / m200 - 1) if m200 else None,
        "rsi": r(rsi(c), 1), "macd": r(macd.iloc[-1], 3), "sig": r(sig.iloc[-1], 3), "mh": r(macd.iloc[-1] - sig.iloc[-1], 3),
        "hi": r(hi, 2), "lo": r(lo, 2), "sw": sw, "dhi": r(last / hi - 1), "dlo": r(last / lo - 1),
        "r1": ret(21), "r3": ret(63), "r6": ret(126), "r12": ret(252),
        "cruce": cruce, "rv": rv, "so": so, "sow": sow,
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
            df = yf.download(lote, period="5y", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=True)
        except Exception as e:
            print("lote falló:", e)
            continue
        # el pedido de 5 años a veces llega sin la última rueda: se completa con uno corto de los últimos días
        try:
            df2 = yf.download(lote, period="5d", interval="1d", group_by="ticker", progress=False, threads=True, auto_adjust=True)
            df = df.combine_first(df2)
        except Exception as e:
            print("completar últimos días falló:", e)
        for s in lote:
            try:
                x = df[s] if len(lote) > 1 else df
                res = calc(x["Close"], x["Volume"] if "Volume" in x else None, x["High"] if "High" in x else None, x["Low"] if "Low" in x else None)
                if res:
                    out[inv[s]] = res
            except Exception:
                pass
    faltan = [t for t in tickers if t not in out]
    for t in faltan[:40]:   # los que no salieron en el pedido grande se piden de a uno
        try:
            x = yf.Ticker(sym[t]).history(period="5y", interval="1d", auto_adjust=True)
            if x.index.tz is not None:
                x.index = x.index.tz_localize(None)
            res = calc(x["Close"], x.get("Volume"), x.get("High"), x.get("Low"))
            if res:
                out[t] = res
        except Exception:
            pass
    if len(out) < len(tickers) * 0.5:
        raise SystemExit(f"Solo {len(out)} de {len(tickers)}: no se guarda para no pisar los datos")
    for t, v in previo.items():   # lo que no se pudo bajar hoy conserva el dato anterior
        out.setdefault(t, v)
    json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "t": out}, open("data/tecnico.json", "w"), separators=(",", ":"))
    import collections
    print(f"{len(out)} de {len(tickers)} con indicadores; última rueda:", collections.Counter(v["wend"] for v in out.values()).most_common(3))


if __name__ == "__main__":
    main()
