"""Investigación: planes de salida más cortos para los círculos verdes (que la plata se mueva más rápido), 5 años, CEDEARs.
Misma entrada (apertura siguiente a la señal), mismo stop inicial (mínimo de 20 ruedas) y misma forma de medir el capital que investigacion_circulos_filtros.py.
Variantes: tope de ruedas, stop que sube, vender todo en un objetivo. Se muestran para la versión sin filtro y para SPY>SMA200 + SMA50>SMA200. Resultado: data/investigacion_circulos_planes.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F

K = F.K
# (nombre, tope de ruedas, modo, múltiplo del objetivo, ruedas del stop que sube)
PLANES = [("Actual: mitad a +2R, stop a la entrada, máx. 60", 60, "mitad", 2.0, 0), ("Igual, máx. 40 ruedas", 40, "mitad", 2.0, 0), ("Igual, máx. 20 ruedas", 20, "mitad", 2.0, 0), ("Igual, máx. 10 ruedas", 10, "mitad", 2.0, 0),
          ("Stop que sube (mín. de 10 ruedas), máx. 60", 60, "trail", 0, 10), ("Stop que sube (mín. de 5 ruedas), máx. 60", 60, "trail", 0, 5),
          ("Todo a +1R (o stop), máx. 20", 20, "todo", 1.0, 0), ("Todo a +2R (o stop), máx. 20", 20, "todo", 2.0, 0), ("Todo a +2R (o stop), máx. 40", 40, "todo", 2.0, 0)]


def camino(P, e, stop, maxr, modo, objm, trail):
    ent = P["O"][e]; R = ent - stop; obj = ent + objm * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []; hk = None
    for j in range(e, min(n, e + maxr)):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if trail and j > e: sc = max(sc, float(np.min(P["L"][max(e, j - trail):j])))
        if lo <= sc:
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path, ((hk, vm) if mitad else None)
        if modo == "todo" and hi >= obj:
            path.append(max(obj, o) / ent * K); return path, None
        if modo == "mitad" and not mitad and hi >= obj:
            vm = max(obj, o) / ent * K; mitad = True; sc = ent; hk = j - e
        v = c / ent * K
        path.append(0.5 * vm + 0.5 * v if mitad else v)
    return path, ((hk, vm) if mitad else None)


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]; spy_ok_s = (sx["Close"] > sx["Close"].rolling(200).mean())
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios)
    dias = [x for x in sx.index if ini <= x <= ult]
    sb_acu, sb_dd, sb_por = F.spy_bh(ini, ult, ini_dl)
    filas = {p[0]: [] for p in PLANES}
    for s, P in datos.items():
        ok = spy_ok_s.reindex(P["f"], method="ffill").fillna(False).values; n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            fl = (1 if ok[i] else 0) | (2 if P["sma50"][i] > P["sma200"][i] else 0) | (4 if P["atrp"][i] < 0.04 else 0)
            for nombre, maxr, modo, objm, trail in PLANES:
                path, half = camino(P, e, stop, maxr, modo, objm, trail); ret = path[-1] - 1; riesgo = (ent - stop) / ent
                filas[nombre].append(dict(s=s, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=fl, path=path, half=half, dts=list(P["f"][e:e + len(path)])))
    L = [f"PLANES DE SALIDA MÁS CORTOS PARA LOS CÍRCULOS VERDES, últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Entrada: apertura siguiente. Stop inicial: mínimo de 20 ruedas. Costo {100 * I.COSTO:.2f}% por lado.",
         "Capital: 1% de riesgo por operación, sin apalancamiento (exposición máx. 100%), se actualiza al cerrar. 'Efectivo/mes' = capital que vuelve a efectivo por mes (puede pasar de 100% porque se reinvierte). 'Meses sin ganancia' = racha máxima de meses seguidos con ganancia realizada ≤ 0.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%.", ""]
    enc = f"{'Plan':<50}| {'ops':>4} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'acum.':>8} | {'caída':>7} | {'c. MTM':>7} | {'rued.':>5} | {'≤10':>4} | {'efec./mes':>9} | {'gan./mes':>8} | {'med./mes':>8} | {'peor mes':>8} | {'mes+':>4} | {'sin gan.':>8}"
    for titulo, m in (("SIN FILTRO", 0), ("SPY>SMA200 + SMA50>SMA200", 3)):
        L += [f"== {titulo} ==", enc]
        for nombre, *_ in PLANES:
            df = pd.DataFrame(filas[nombre]); df = df[(df["fl"] & m) == m]
            res = F.sim(df); c = F.caja_datos(res, dias); mm = F.mercado(res, dias)
            r = df["ret"].values; R = df["rnet"].values
            L.append(f"{nombre:<50}| {res['tom']:>4} | {100 * r.mean():+6.2f}% | {100 * np.median(r):+6.2f}% | {100 * (r > 0).mean():4.0f}% | {R.mean():+5.2f} | {100 * res['acu']:+7.1f}% | {100 * res['dd']:+6.1f}% | {100 * mm[0]:+6.1f}% | {np.median(c['dur']):>5.0f} | {100 * (c['dur'] <= 10).mean():3.0f}% | {100 * c['inflow'].mean():8.0f}% | {100 * c['pnl'].mean():+7.2f}% | {100 * np.median(c['pnl']):+7.2f}% | {100 * c['pnl'].min():+7.1f}% | {100 * (c['pnl'] > 0).mean():3.0f}% | {c['racha_m']:>8}")
        L.append("")
    L.append("rued. = mediana de ruedas hasta cerrar; ≤10 = % de operaciones cerradas en 10 ruedas o menos; caída = solo operaciones cerradas; c. MTM = valuando las abiertas a mercado.")
    txt = "\n".join(L); open("data/investigacion_circulos_planes.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
