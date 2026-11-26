"""Investigación: ¿la regla de las 20 ruedas funciona igual según la volatilidad de la acción? 5 años, CEDEARs, sin filtro.
Mismas convenciones que investigacion_circulos_20.py: entrada en la apertura siguiente, costo 0,15% por lado, stop inicial = mínimo de 20 ruedas,
stop antes que objetivo en la misma vela, gaps al open, mitad a +2R y stop al precio de compra; capital con 1% de riesgo por operación, exposición máx. 100%.
Grupos por ATR % el día de la señal (promedio de 14 ruedas del rango verdadero / cierre, igual que el dashboard): baja < 2%, media 2% a 4%, alta > 4%.
Versiones: A (cerrar todo en la rueda N), D (en la rueda N: si está en +1R o más, sigue con stop en el precio de compra hasta la rueda 2N; si no, cierra), E (sin límite, máx. 60).
N = 20 (principal) y también 10 y 30 para A y D. Cada grupo se simula con su propio capital (solo opera las señales de ese grupo).
Resultado: data/investigacion_circulos_20_vol.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F

K = F.K
GRUPOS = [("Baja (ATR % < 2%)", 0.0, 0.02), ("Media (ATR % 2% a 4%)", 0.02, 0.04), ("Alta (ATR % > 4%)", 0.04, 9.0)]
MIN_OPS = 200


def camino(P, e, stop, dec, regla, tope_sigue, tope):
    """Valor de la operación (por unidad invertida, neto de costos) al cierre de cada rueda. regla: None (cierra en `tope`) o "1R" (decide en la rueda `dec`)."""
    ent = P["O"][e]; R = ent - stop; obj = ent + 2 * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []; hk = None
    fin = e + tope; siguio = False; j = e
    while j < min(n, fin):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if lo <= sc:
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path, ((hk, vm) if mitad else None), siguio
        if not mitad and hi >= obj:
            vm = max(obj, o) / ent * K; mitad = True; sc = ent; hk = j - e
        path.append(0.5 * vm + 0.5 * (c / ent * K) if mitad else c / ent * K)
        if regla and j - e + 1 == dec:
            if c >= ent + R:
                siguio = True; fin = e + tope_sigue; sc = max(sc, ent)
            else:
                return path, ((hk, vm) if mitad else None), siguio
        j += 1
    return path, ((hk, vm) if mitad else None), siguio


def llega(P, e, stop, ruedas=20):
    """¿Llega a +1R y a +2R (máximo de la vela) dentro de las primeras `ruedas` ruedas, antes de tocar el stop inicial? (stop primero en la misma vela)"""
    ent = P["O"][e]; R = ent - stop; n = len(P["C"]); a1 = a2 = False
    for j in range(e, min(n, e + ruedas)):
        if P["L"][j] <= stop: break
        if P["H"][j] >= ent + R: a1 = True
        if P["H"][j] >= ent + 2 * R: a2 = True; break
    return a1, a2


# (clave, nombre, dec, regla, tope si sigue, tope)
def versiones(dec):
    return [(f"A{dec}", f"A. Cierra todo en la rueda {dec}", dec, None, dec, dec), (f"D{dec}", f"D. En la {dec}: si está en +1R sigue hasta la {2 * dec}", dec, "1R", 2 * dec, dec)]


VERS = versiones(20) + [("E", "E. Referencia: sin límite (máx. 60)", 20, None, 60, 60)] + versiones(10) + versiones(30)


def metricas(df, dias):
    if not len(df): return None
    res = F.sim(df); c = F.caja_datos(res, dias); mm = F.mercado(res, dias)
    r, R = df["ret"].values, df["rnet"].values
    return dict(res=res, ops=len(df), tom=res["tom"], prom=r.mean(), med=np.median(r), gan=(r > 0).mean(), R=R.mean(), dur=np.median(df["dur"].values),
                acu=res["acu"], mtm=mm[0], efec=c["inflow"].mean() if c else np.nan, mesp=(c["pnl"] > 0).mean() if c else np.nan, peor=c["pnl"].min() if c else np.nan)


def linea(nombre, x, marca=""):
    if x is None: return f"{nombre:<46}| sin operaciones"
    return (f"{nombre:<46}| {x['ops']:>4}{marca:<1} | {x['tom']:>5} | {100 * x['prom']:+6.2f}% | {100 * x['med']:+6.2f}% | {100 * x['gan']:3.0f}% | {x['R']:+5.2f} | {x['dur']:>4.0f} | "
            f"{100 * x['acu']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {100 * x['efec']:8.0f}% | {100 * x['mesp']:3.0f}% | {100 * x['peor']:+7.1f}%")


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    filas = {v[0]: [] for v in VERS}; alc = []
    for s, P in datos.items():
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            riesgo = (ent - stop) / ent; atrp = float(P["atrp"][i])
            a1, a2 = llega(P, e, stop); alc.append(dict(s=s, atrp=atrp, a1=a1, a2=a2, riesgo=riesgo))
            for clave, _, dec, regla, tope_sigue, tope in VERS:
                path, half, siguio = camino(P, e, stop, dec, regla, tope_sigue, tope); ret = path[-1] - 1
                filas[clave].append(dict(s=s, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0,
                                         path=path, half=half, dts=list(P["f"][e:e + len(path)]), dur=len(path), siguio=siguio, atrp=atrp))
    DF = {k: pd.DataFrame(v) for k, v in filas.items()}; AL = pd.DataFrame(alc)
    L = [f"¿LA REGLA DE LAS 20 RUEDAS FUNCIONA IGUAL SEGÚN LA VOLATILIDAD? Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Sin filtro.",
         f"Plan base: entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra. Costo {100 * I.COSTO:.2f}% por lado. Stop antes que objetivo en la misma vela; gaps al open.",
         "Grupos por ATR % el día de la señal (el mismo que muestra el dashboard). Cada grupo se simula con su propio capital: 1% de riesgo por operación, sin apalancamiento (exposición máx. 100%).",
         "A = cierra todo en la rueda N. D = en la rueda N, si está en +1R o más sigue con stop en el precio de compra hasta la rueda 2N; si no, cierra. E = sin límite (máx. 60). La rueda 1 es la de entrada.",
         "Columnas: ops = señales del grupo (* = menos de 200: poco confiable); toma. = operaciones tomadas con el capital; dur. = mediana de ruedas; acum. = acumulado del capital;",
         "c. MTM = peor caída valuando las abiertas a mercado; efec./mes = capital que vuelve a efectivo por mes; mes+ = % de meses con ganancia realizada; peor mes = peor mes realizado.",
         "LÍMITES: se probaron varias combinaciones (3 grupos × 7 versiones) y se elige la mejor (sesgo de selección); la lista de CEDEARs es la de hoy (sesgo de supervivencia);",
         "cada grupo tiene menos operaciones que el total, así que sus resultados son menos firmes; las operaciones se superponen y no son independientes.", ""]
    L.append("== TAMAÑO DE CADA GRUPO ==")
    for g, lo, hi in GRUPOS:
        a = AL[(AL["atrp"] >= lo) & (AL["atrp"] < hi)]
        acc = a["s"].value_counts()
        L.append(f"{g:<24}: {len(a):>5} señales en {acc.size:>3} acciones{'  ← MENOS DE 200: POCO CONFIABLE' if len(a) < MIN_OPS else ''}. Más señales: {', '.join(f'{k} ({v})' for k, v in acc.head(8).items())}. "
                 f"Distancia mediana al stop: {100 * a['riesgo'].median():.1f}%.")
    L.append("")
    L.append("== ¿CUÁNTAS LLEGAN A +1R Y A +2R DENTRO DE LAS PRIMERAS 20 RUEDAS (antes de tocar el stop inicial)? ==")
    for g, lo, hi in GRUPOS:
        a = AL[(AL["atrp"] >= lo) & (AL["atrp"] < hi)]
        if len(a): L.append(f"{g:<24}: llegan a +1R {100 * a['a1'].mean():4.0f}% | llegan a +2R (venden la mitad) {100 * a['a2'].mean():4.0f}% | {len(a)} operaciones")
    L.append("")
    enc = f"{'Versión':<46}| {'ops':>5} | {'toma.':>5} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'dur.':>4} | {'acum.':>8} | {'c. MTM':>7} | {'efec./mes':>9} | {'mes+':>4} | {'peor mes':>8}"
    mejor = {}
    for g, lo, hi in GRUPOS:
        L += [f"== GRUPO {g.upper()} ==", enc]
        res_g = {}
        for clave, nombre, *_ in VERS:
            df = DF[clave]; sub = df[(df["atrp"] >= lo) & (df["atrp"] < hi)]
            x = metricas(sub, dias); res_g[clave] = (sub, x)
            if clave == "A10": L.append("-- probando otra rueda de decisión (10 y 30) --")
            L.append(linea(nombre, x, "*" if x and x["ops"] < MIN_OPS else ""))
        # después de la rueda N en D
        for clave in ("D10", "D20", "D30"):
            sub = res_g[clave][0]
            if not len(sub): continue
            dec = int(clave[1:]); gs = sub[sub["siguio"]]
            if len(gs):
                post = np.array([p[-1] - p[dec - 1] for p in gs["path"]])
                L.append(f"   {clave}: siguieron después de la {dec} {len(gs)} de {len(sub)} ({100 * len(gs) / len(sub):.0f}%); después de la {dec} ganaron en promedio {100 * post.mean():+.2f}% (mediana {100 * np.median(post):+.2f}%), mejoraron {100 * (post > 0).mean():.0f}%.")
            else:
                L.append(f"   {clave}: ninguna llegó a +1R en la rueda {dec}, así que D es igual a A.")
        # mitades para A20, D20, E
        L.append(f"-- por mitades (señales hasta {corte.date()} | desde {corte.date()}; el capital vuelve a empezar en cada mitad) --")
        rk1, rk2 = [], []
        for clave in ("A20", "D20", "E"):
            sub = res_g[clave][0]; a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
            xa, xb = metricas(a, d1), metricas(b, d2)
            fa = (f"{len(a):>4}{'*' if len(a) < MIN_OPS else ' '} ops, acum. {100 * xa['acu']:+7.1f}%, R {xa['R']:+5.2f}, c. MTM {100 * xa['mtm']:+6.1f}%, mes+ {100 * xa['mesp']:3.0f}%") if xa else "sin operaciones"
            fb = (f"{len(b):>4}{'*' if len(b) < MIN_OPS else ' '} ops, acum. {100 * xb['acu']:+7.1f}%, R {xb['R']:+5.2f}, c. MTM {100 * xb['mtm']:+6.1f}%, mes+ {100 * xb['mesp']:3.0f}%") if xb else "sin operaciones"
            L.append(f"   {clave:<4}| 1.ª: {fa} || 2.ª: {fb}")
            if xa: rk1.append((clave, xa["acu"]))
            if xb: rk2.append((clave, xb["acu"]))
        L.append(f"   Orden por acumulado: 1.ª mitad {' > '.join(k for k, _ in sorted(rk1, key=lambda t: -t[1]))} | 2.ª mitad {' > '.join(k for k, _ in sorted(rk2, key=lambda t: -t[1]))}")
        tot = [(k, res_g[k][1]["acu"]) for k in ("A20", "D20", "E") if res_g[k][1]]
        mejor[g] = sorted(tot, key=lambda t: -t[1])[0][0] if tot else "—"
        L.append("")
    L.append("== RESUMEN: MEJOR VERSIÓN (A20, D20 o E) POR ACUMULADO EN 5 AÑOS ==")
    for g, *_ in GRUPOS: L.append(f"{g:<24}: {mejor[g]}")
    L.append("(* = grupo o mitad con menos de 200 operaciones: poco confiable.)")
    txt = "\n".join(L); open("data/investigacion_circulos_20_vol.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
