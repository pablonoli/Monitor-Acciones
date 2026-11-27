"""Investigación (para jugar en papel): ¿conviene ponerse en corto en BTC con velas de 15 minutos cuando BTC está en tendencia bajista?
Datos: velas de 15 minutos de BTCUSDT (Binance, data-api.binance.vision) desde enero 2024.
Tendencia del día: bajista si el cierre diario de BTC está debajo de su media de 200 días (se usa el cierre del día anterior: no mira el futuro).
Señal de corto ("espejo" del círculo verde): el cierre de 15 m rompe el piso de las 20 velas anteriores, el cierre anterior estaba arriba de ese piso
y la caída es fuerte (Z del movimiento < -1,5 respecto de las últimas 50 velas). Señal de largo: lo mismo para arriba (Z > 1,5).
Plan (igual que el de las acciones, en velas de 15 m): entrada en la apertura de la vela siguiente; stop en el máximo de las 20 velas (corto) o el mínimo
(largo); mitad a 2R y stop al precio de entrada; cierre de todo a las 20 velas (5 horas). Stop antes que objetivo en la misma vela; saltos al open.
Costo 0,07% por lado (comisión de futuros + deslizamiento); no incluye la tasa de financiamiento. Una operación a la vez.
Se saltean las señales con el stop a menos de 0,4% o a más de 5% de la entrada (para todas las versiones, también la de azar).
Resultados en R (1R = lo que se arriesga hasta el stop). Base de comparación: cortos al azar con el mismo plan en los días bajistas.
Resultado: data/investigacion_btc_corto_15m.txt"""
import json, sys, time, urllib.request
import numpy as np, pandas as pd

COSTO = 0.0007
URL = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval={iv}&limit=1000&startTime={t}"


def bajar(iv, desde):
    t = int(pd.Timestamp(desde).timestamp() * 1000); filas = []
    while True:
        for intento in range(4):
            try:
                with urllib.request.urlopen(URL.format(iv=iv, t=t), timeout=30) as r: lote = json.load(r); break
            except Exception as e:
                print("reintento", iv, t, e); time.sleep(3)
        else:
            raise RuntimeError("no se pudo bajar " + iv)
        if not lote: break
        filas += lote; t = lote[-1][0] + 1
        if len(lote) < 1000: break
        time.sleep(0.15)
    x = pd.DataFrame(filas, columns=list("tohlcv") + ["T", "q", "n", "a", "b", "i"])[list("tohlcv")].astype(float)
    x.index = pd.to_datetime(x["t"], unit="ms"); return x[["o", "h", "l", "c", "v"]]


def senales(x, lado, Zmin=1.5, N=20):
    c, h, l = x["c"], x["h"], x["l"]; mom = c.diff(); z = (mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)
    if lado == "corto":
        piso = l.shift(1).rolling(N).min(); return ((c < piso) & (c.shift(1) >= piso.shift(1)) & (z < -Zmin)).values
    techo = h.shift(1).rolling(N).max(); return ((c > techo) & (c.shift(1) <= techo.shift(1)) & (z > Zmin)).values


def operar(x, sen, lado, permitido, N=20, tope=20, rng=None):
    O, H, L, C, idx, n = x["o"].values, x["h"].values, x["l"].values, x["c"].values, x.index, len(x)
    alto = x["h"].rolling(N).max().values; bajo = x["l"].rolling(N).min().values
    out = []; i = N + 50
    while i < n - 1:
        if not (sen[i] and permitido[i]): i += 1; continue
        e = i + 1; ent = O[e]
        if lado == "corto":
            st = alto[i]; R = st - ent
        else:
            st = bajo[i]; R = ent - st
        if not (R > 0) or R / ent > 0.05 or R / ent < 0.004: i += 1; continue   # stop entre 0,4% y 5%: con stops más cortos la comisión se come el resultado
        s = 1 if lado == "largo" else -1; obj = ent + s * 2 * R; sc = st; mitad = None; res = None; j = e
        while j < min(n, e + tope):
            toca = (L[j] <= sc) if s == 1 else (H[j] >= sc)
            if toca:
                px = min(sc, O[j]) if s == 1 else max(sc, O[j]); res = (px, j); break
            alc = (H[j] >= obj) if s == 1 else (L[j] <= obj)
            if mitad is None and alc: mitad = max(obj, O[j]) if s == 1 else min(obj, O[j]); sc = ent
            j += 1
        if res is None: j = min(n, e + tope) - 1; res = (C[j], j)
        px, j = res
        bruto = lambda p: s * (p - ent) / R
        r = (0.5 * bruto(mitad) + 0.5 * bruto(px)) if mitad is not None else bruto(px)
        r -= 2 * COSTO * ent / R
        out.append(dict(t=idx[i], r=r, riesgo=R / ent, dur=j - e + 1))
        i = j + 1
    return pd.DataFrame(out)


def resumen(df, nombre):
    if df is None or not len(df): return f"{nombre:<52} sin operaciones"
    r = df["r"].values; eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    return (f"{nombre:<52} {len(r):>5} op. · {100 * (r > 0).mean():3.0f}% ganadoras · promedio {r.mean():+.3f}R · total {r.sum():+7.1f}R · "
            f"peor racha {dd:+6.1f}R · stop medio {100 * df['riesgo'].mean():.2f}%")


def main(iv="15m"):
    x = bajar(iv, "2023-09-01"); d = bajar("1d", "2023-01-01")
    sma = d["c"].rolling(200).mean(); baj_d = (d["c"] < sma).shift(1)   # tendencia de ayer
    dia = x.index.normalize(); bajista = baj_d.reindex(dia).fillna(False).values.astype(bool); alcista = ~bajista & ~np.isnan(sma.shift(1).reindex(dia).values)
    x = x[x.index >= "2024-01-01"]; bajista = bajista[-len(x):]; alcista = alcista[-len(x):]
    sc, sl = senales(x, "corto"), senales(x, "largo"); todo = np.ones(len(x), bool)
    corte = x.index[len(x) // 2]
    hoy_baj = bool(d["c"].iloc[-1] < sma.iloc[-1])
    HORAS = {"15m": "5 horas", "1h": "20 horas", "4h": "unos 3 días"}
    L = [f"BTC VELAS DE {iv}: ¿CORTO EN TENDENCIA BAJISTA? (cierre a las 20 velas = {HORAS.get(iv, '20 velas')}) Velas de {x.index[0].date()} a {x.index[-1]:%d/%m/%Y %H:%M} UTC ({len(x)} velas).",
         f"Hoy BTC cerró en {d['c'].iloc[-1]:,.0f} con media de 200 días en {sma.iloc[-1]:,.0f}: tendencia {'BAJISTA' if hoy_baj else 'ALCISTA'}.".replace(",", "."),
         f"Días bajistas en la muestra: {100 * bajista.mean():.0f}% del tiempo.",
         "Costo 0,07% por lado, sin tasa de financiamiento. Resultados en R por operación (1R = la pérdida si toca el stop).",
         "LÍMITES: un solo activo, un período (2024-2026) con pocos tramos bajistas largos; varias versiones probadas; el resultado puede ser casualidad.", ""]
    rng = np.random.default_rng(7)
    azar = rng.random(len(x)) < sc.mean() * 3
    V = [("Corto, solo en días bajistas (la idea)", sc, "corto", bajista),
         ("Corto, siempre (sin mirar la tendencia)", sc, "corto", todo),
         ("Corto en días ALCISTAS (contra la tendencia)", sc, "corto", alcista),
         ("Corto AL AZAR en días bajistas (base)", azar, "corto", bajista),
         ("Largo, solo en días alcistas", sl, "largo", alcista),
         ("Largo, en días bajistas (contra la tendencia)", sl, "largo", bajista)]
    L.append("== RESULTADOS (2024 a hoy) ==")
    R = {}
    for nom, s_, lado, perm in V:
        df = operar(x, s_, lado, perm); R[nom] = df; L.append(resumen(df, nom))
    L += ["", f"== POR MITADES (antes y después del {corte:%d/%m/%Y}) =="]
    for nom, df in R.items():
        if not len(df): continue
        a, b = df[df["t"] < corte], df[df["t"] >= corte]
        L.append(f"{nom:<52} 1.ª: {len(a):>4} op., {a['r'].mean() if len(a) else 0:+.3f}R prom., total {a['r'].sum():+6.1f}R · 2.ª: {len(b):>4} op., {b['r'].mean() if len(b) else 0:+.3f}R prom., total {b['r'].sum():+6.1f}R")
    L += ["", "== CORTO EN DÍAS BAJISTAS, POR AÑO =="]
    df = R[V[0][0]]
    for a, g in df.groupby(df["t"].dt.year): L.append(f"{a}: {len(g)} op. · {100 * (g['r'] > 0).mean():.0f}% ganadoras · total {g['r'].sum():+.1f}R")
    L += ["", "== ÚLTIMAS SEÑALES DE CORTO (días bajistas) =="]
    for _, f in df.tail(8).iterrows(): L.append(f"{f['t']:%d/%m %H:%M} UTC · {f['r']:+.2f}R")
    txt = "\n".join(L); open(f"data/investigacion_btc_corto_{iv}.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    for iv in (sys.argv[1:] or ["15m"]): main(iv)
