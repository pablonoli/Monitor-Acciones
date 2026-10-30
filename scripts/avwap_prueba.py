"""Prueba de cómo funciona el AVWAP (VWAP anclado) como soporte y resistencia, con 10 acciones de las que tienen CEDEAR.
Para cada rueda se usa solo la información que había hasta ese día (sin mirar el futuro) y después se mide qué pasó a 20 y 60 ruedas.
Compara contra el promedio de todas las ruedas y contra la misma regla con la media de 200 ruedas.
Resultado: data/avwap_prueba.json y data/avwap_prueba.md (solo para leer, el sitio no los usa)."""
import json, math, sys
import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "NVDA", "MELI", "KO", "WMT", "JPM", "XOM", "DIS", "PFE"]
LOOK, MINB, SEP, H1, H2 = 252, 40, 10, 20, 60


def episodios(idx, sep=SEP):
    """de una lista de ruedas que cumplen la regla, deja la primera de cada racha (separadas por al menos `sep` ruedas)"""
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= sep:
            out.append(i)
        last = i
    return out


def resumen(rows):
    """rows: lista de dicts con f20, f60, held"""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    f20 = np.array([r["f20"] for r in rows if r["f20"] is not None], float)
    f60 = np.array([r["f60"] for r in rows if r["f60"] is not None], float)
    hl = [r["held"] for r in rows if r.get("held") is not None]
    e20 = np.array([r["e20"] for r in rows if r.get("e20") is not None], float)
    e60 = np.array([r["e60"] for r in rows if r.get("e60") is not None], float)
    o = {"n": n}
    if len(f20):
        o.update(m20=round(float(f20.mean()) * 100, 2), md20=round(float(np.median(f20)) * 100, 2), p20=round(float((f20 > 0).mean()) * 100, 1),
                 se20=round(float(f20.std(ddof=1) / math.sqrt(len(f20))) * 100, 2) if len(f20) > 2 else None)
    if len(f60):
        o.update(m60=round(float(f60.mean()) * 100, 2), md60=round(float(np.median(f60)) * 100, 2), p60=round(float((f60 > 0).mean()) * 100, 1), n60=len(f60))
    if len(e20) > 2:
        o.update(x20=round(float(e20.mean()) * 100, 2), xse20=round(float(e20.std(ddof=1) / math.sqrt(len(e20))) * 100, 2))
    if len(e60) > 2:
        o.update(x60=round(float(e60.mean()) * 100, 2), xse60=round(float(e60.std(ddof=1) / math.sqrt(len(e60))) * 100, 2))
    if hl:
        o["held"] = round(float(np.mean(hl)) * 100, 1)
    return o


def analizar(df):
    """df: columnas High, Low, Close, Volume con índice de fechas. Devuelve {regla: [eventos]} y la base de todas las ruedas."""
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    n = len(df)
    if n < LOOK + H2 + 60:
        return None
    c, hi, lo, v = (df[k].values.astype(float) for k in ["Close", "High", "Low", "Volume"])
    tp = (hi + lo + c) / 3
    PV, V = np.cumsum(tp * v), np.cumsum(v)

    def avwap_path(a):   # AVWAP anclado en la rueda a, para todas las ruedas desde a
        j = np.arange(a, n)
        return (PV[j] - (PV[a - 1] if a > 0 else 0)) / (V[j] - (V[a - 1] if a > 0 else 0))
    cache = {}

    def path(a):
        if a not in cache:
            cache[a] = avwap_path(a)
        return cache[a]
    sma = pd.Series(c).rolling(200).mean().values

    def fwd(t, h):
        return c[t + h] / c[t] - 1 if t + h < n else None

    def ev(t, nivel, tipo):
        e = {"f20": fwd(t, H1), "f60": fwd(t, H2)}
        if t + H1 < n:
            if tipo == "sop":   # aguantó: en las 20 ruedas siguientes el mínimo no perforó el nivel más de 3%
                e["held"] = bool(lo[t + 1:t + H1 + 1].min() > nivel * 0.97)
            else:               # frenó: en las 20 ruedas siguientes el máximo no pasó el nivel más de 2%
                e["held"] = bool(hi[t + 1:t + H1 + 1].max() < nivel * 1.02)
        return e

    sup, res, rup, s200, base = [], [], [], [], []
    ts_sup, ts_res, ts_rup, ts_200 = [], [], [], []
    for t in range(LOOK, n - 1):
        if t + H1 < n:
            base.append({"f20": fwd(t, H1), "f60": fwd(t, H2)})
        w = slice(t - LOOK + 1, t + 1)
        # --- soporte: AVWAP desde el mínimo de 52 semanas
        a = t - LOOK + 1 + int(np.argmin(lo[w]))
        if t - a >= MINB:
            p = path(a)[: t - a + 1]
            rel = c[a + 1:t + 1] / p[1:] - 1
            d = c[t] / p[-1] - 1
            if (rel > 0).mean() >= 0.75 and rel.max() >= 0.05 and -0.015 <= d <= 0.03:
                ts_sup.append(t)
        # --- resistencia: AVWAP desde el máximo de 52 semanas
        b = t - LOOK + 1 + int(np.argmax(hi[w]))
        if t - b >= MINB:
            p = path(b)[: t - b + 1]
            rel = c[b + 1:t + 1] / p[1:] - 1
            d = c[t] / p[-1] - 1
            if (rel < 0).mean() >= 0.75 and rel.min() <= -0.08:
                if -0.03 <= d <= 0.005:
                    ts_res.append(t)
                prev = c[t - 1] / path(b)[t - 1 - b] - 1 if t - 1 > b else None
                if prev is not None and prev < 0.02 <= d and (rel[:-1] < 0).mean() >= 0.75:
                    ts_rup.append(t)
        # --- control: la misma regla de soporte con la media de 200 ruedas
        if not math.isnan(sma[t]) and t >= 200 + 120:
            rr = c[t - 119:t + 1] / sma[t - 119:t + 1] - 1
            d = rr[-1]
            if (rr > 0).mean() >= 0.75 and rr.max() >= 0.08 and -0.015 <= d <= 0.03:
                ts_200.append(t)
    # eventos con su nivel (se recalcula el nivel de cada día de evento)
    def nivel_sup(t):
        a = t - LOOK + 1 + int(np.argmin(lo[t - LOOK + 1:t + 1])); return path(a)[t - a]

    def nivel_res(t):
        b = t - LOOK + 1 + int(np.argmax(hi[t - LOOK + 1:t + 1])); return path(b)[t - b]
    out = {
        "soporte_avwap_min": [ev(t, nivel_sup(t), "sop") for t in episodios(ts_sup)],
        "soporte_media200": [ev(t, sma[t], "sop") for t in episodios(ts_200)],
        "resistencia_avwap_max": [ev(t, nivel_res(t), "res") for t in episodios(ts_res)],
        "ruptura_avwap_max": [ev(t, nivel_res(t), "sop") for t in episodios(ts_rup)],
    }
    return out, base


def main():
    import yfinance as yf
    tickers, sufijo = TICKERS, ""
    if "--100" in sys.argv:   # 100 empresas con CEDEAR repartidas parejo (de la lista ordenada), sin las que reportan en pesos
        co = json.load(open("data/datos.json"))["co"]
        uni = sorted(u["t"] for u in json.load(open("data/universe.json"))["list"] if u.get("k") == "CEDEAR")
        uni = [t for t in uni if co.get(t, {}).get("status") == "ok" and co[t].get("cur") != "ARS" and "-" not in t]
        tickers, sufijo = [uni[int(i)] for i in np.linspace(0, len(uni) - 1, 100)], "_100"
    todo, base_all, por = {}, [], {}
    for tk in tickers:
        try:
            df = yf.download(tk, period="10y", interval="1d", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            res = analizar(df)
        except Exception as e:
            print(tk, "falló:", e)
            continue
        if not res:
            print(tk, "sin datos suficientes")
            continue
        ev, base = res
        b20 = np.mean([x["f20"] for x in base if x["f20"] is not None])
        b60 = np.mean([x["f60"] for x in base if x["f60"] is not None])
        for x in sum(ev.values(), []):   # exceso sobre el promedio de esa misma acción (para no premiar acciones que subieron mucho)
            x["e20"] = None if x["f20"] is None else x["f20"] - b20
            x["e60"] = None if x["f60"] is None else x["f60"] - b60
        base_all += base
        por[tk] = {k: resumen(x) for k, x in ev.items()}
        por[tk]["todas_las_ruedas"] = resumen(base)
        for k, x in ev.items():
            todo.setdefault(k, []).extend(x)
        print(tk, {k: len(x) for k, x in ev.items()})
    pool = {k: resumen(x) for k, x in todo.items()}
    pool["todas_las_ruedas"] = resumen(base_all)
    json.dump({"tickers": list(por), "pool": pool, "por_accion": por}, open(f"data/avwap_prueba{sufijo}.json", "w"), indent=1, ensure_ascii=False)
    nom = {"soporte_avwap_min": "Soporte: AVWAP desde el mínimo de 52 semanas (precio tocándolo desde arriba)",
           "soporte_media200": "Control: la misma regla con la media de 200 ruedas",
           "resistencia_avwap_max": "Resistencia: AVWAP desde el máximo de 52 semanas (precio tocándolo desde abajo)",
           "ruptura_avwap_max": "Ruptura: el precio supera ese AVWAP en más de 2%",
           "todas_las_ruedas": "Base: cualquier rueda"}
    L = ["# Prueba del AVWAP", "", f"{len(por)} acciones: {', '.join(por)}. Datos de 10 años, una señal por racha (separadas por al menos {SEP} ruedas). Retornos a {H1} y {H2} ruedas, en %.", "",
         "| Regla | Señales | Prom. 20r | % positivas 20r | Exceso vs la misma acción 20r (± error) | Prom. 60r | % positivas 60r | Exceso 60r (± error) | Aguantó/frenó |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ["soporte_avwap_min", "soporte_media200", "resistencia_avwap_max", "ruptura_avwap_max", "todas_las_ruedas"]:
        s = pool.get(k, {"n": 0})
        L.append(f"| {nom[k]} | {s['n']} | {s.get('m20','—')} | {s.get('p20','—')} | {s.get('x20','—')} (± {s.get('xse20','—')}) | {s.get('m60','—')} | {s.get('p60','—')} | {s.get('x60','—')} (± {s.get('xse60','—')}) | {s.get('held','—')} |")
    L += ["", "## Por acción (señales de soporte AVWAP / control media 200 / resistencia)", "", "| Acción | Soporte AVWAP (n, prom 20r, % pos) | Media 200 (n, prom 20r, % pos) | Resistencia (n, prom 20r, % pos) | Base prom 20r |", "|---|---|---|---|---|"]
    for tk, d in por.items():
        f = lambda s: f"{s['n']}, {s.get('m20','—')}, {s.get('p20','—')}" if s["n"] else "0"
        L.append(f"| {tk} | {f(d['soporte_avwap_min'])} | {f(d['soporte_media200'])} | {f(d['resistencia_avwap_max'])} | {d['todas_las_ruedas'].get('m20','—')} |")
    open(f"data/avwap_prueba{sufijo}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
