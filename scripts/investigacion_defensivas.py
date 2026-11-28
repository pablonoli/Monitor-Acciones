"""Investigación: ¿las acciones defensivas se mueven menos en 20 ruedas, y por eso la regla de las 20 ruedas les rinde menos? 5 años, CEDEARs.
Defensivas = consumo básico, salud, servicios públicos, telecomunicaciones e inmobiliarias (lista fija abajo, solo las que tienen CEDEAR).
Compara "defensivas" contra "resto" con las mismas convenciones de investigacion_circulos_20_vol.py: entrada en la apertura siguiente, costo 0,15% por lado,
stop inicial = mínimo de 20 ruedas, stop antes que objetivo en la misma vela, gaps al open, mitad a +2R y stop al precio de compra; capital con 1% de riesgo, exposición máx. 100%.
Resultado: data/investigacion_defensivas.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_20_vol as V
import investigacion_circulos_filtros as F

DEFENSIVAS = "KO PEP PG JNJ MO WMT VZ T MCD HSY PFE O ABBV MRK ABT NEE COST AMGN UNH ABEV CL KMB GIS MDLZ PM BMY GILD CVS DUK SO".split()
VERS = V.versiones(20) + [("E", "E. Referencia: sin límite (máx. 60)", 20, None, 60, 60),
                          ("A30", "A. Cierra todo en la rueda 30", 30, None, 30, 30), ("A40", "A. Cierra todo en la rueda 40", 40, None, 40, 40)]
MIN_OPS = 200


def movimiento(P, ini):
    """Para cada rueda: retorno a 20 ruedas, máxima suba y máxima baja dentro de esas 20 ruedas (en % del cierre de hoy), y si fue señal verde."""
    n = len(P["C"]); out = []
    for i in range(210, n - 20):
        if P["f"][i] < ini: continue
        c = P["C"][i]
        out.append((P["C"][i + 20] / c - 1, max(P["H"][i + 1:i + 21]) / c - 1, min(P["L"][i + 1:i + 21]) / c - 1, bool(P["verde"][i]), float(P["atrp"][i])))
    return out


def resumen_mov(filas, titulo):
    if not filas: return f"{titulo:<34}| sin datos"
    a = np.array([(abs(f[0]), f[1], f[2], f[4]) for f in filas]); r = np.array([f[0] for f in filas])
    return (f"{titulo:<34}| {len(filas):>7} | {100 * np.nanmedian(a[:, 3]):4.1f}% | {100 * np.median(a[:, 0]):4.1f}% | {100 * np.percentile(a[:, 0], 75):4.1f}% | "
            f"{100 * (a[:, 0] >= 0.10).mean():3.0f}% | {100 * np.median(a[:, 1]):+5.1f}% | {100 * np.median(a[:, 2]):+6.1f}% | {100 * (a[:, 1] >= 0.10).mean():3.0f}%")


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    defs = [s for s in DEFENSIVAS if s in datos]; resto = [s for s in datos if s not in defs]
    filas = {v[0]: [] for v in VERS}; alc = []; mov = {"def": [], "resto": []}
    for s, P in datos.items():
        g = "def" if s in defs else "resto"
        mov[g] += movimiento(P, ini)
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            riesgo = (ent - stop) / ent; atrp = float(P["atrp"][i])
            a1, a2 = V.llega(P, e, stop); alc.append(dict(s=s, g=g, atrp=atrp, a1=a1, a2=a2, riesgo=riesgo))
            for clave, _, dec, regla, tope_sigue, tope in VERS:
                path, half, siguio = V.camino(P, e, stop, dec, regla, tope_sigue, tope); ret = path[-1] - 1
                filas[clave].append(dict(s=s, g=g, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0,
                                         path=path, half=half, dts=list(P["f"][e:e + len(path)]), dur=len(path), siguio=siguio, atrp=atrp))
    DF = {k: pd.DataFrame(v) for k, v in filas.items()}; AL = pd.DataFrame(alc)
    L = [f"¿LAS DEFENSIVAS SE MUEVEN MENOS EN 20 RUEDAS? Últimos {anios} años ({ini.date()} a {ult.date()}). Sin filtro.",
         f"Defensivas ({len(defs)}): {', '.join(defs)}.", f"Resto ({len(resto)} acciones con CEDEAR): todas las demás de la lista.",
         f"Plan base: entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra. Costo {100 * I.COSTO:.2f}% por lado. Capital con 1% de riesgo por operación, exposición máx. 100%, cada grupo con su propio capital.",
         "LÍMITES: se probaron varias combinaciones y se elige la mejor (sesgo de selección); la lista de CEDEARs es la de hoy (sesgo de supervivencia); las defensivas son pocas (20) y sus señales también;",
         "las operaciones se superponen y no son independientes; la división defensivas/resto es una clasificación mía por sector, no la lista 'def' del dashboard.", ""]
    L += ["== 1) ¿CUÁNTO SE MUEVEN EN 20 RUEDAS? (todas las ruedas de los últimos años, no solo las señales) ==",
          "ATR % = rango diario típico. |ret 20| = movimiento absoluto del cierre a 20 ruedas (mediana y percentil 75). ≥10% = % de veces que el cierre se movió 10% o más. Sube máx. y baja máx. = mayor suba y mayor baja dentro de las 20 ruedas (mediana). Sube ≥10% = % de veces que en algún momento subió 10% o más.",
          f"{'Grupo':<34}| {'ruedas':>7} | {'ATR %':>5} | {'|r20| med':>9} | {'p75':>5} | {'≥10%':>4} | {'sube máx.':>9} | {'baja máx.':>9} | {'sube≥10%':>8}"]
    L.append(resumen_mov(mov["def"], "Defensivas (todas las ruedas)")); L.append(resumen_mov(mov["resto"], "Resto (todas las ruedas)"))
    L.append(resumen_mov([f for f in mov["def"] if f[3]], "Defensivas (solo días de señal)")); L.append(resumen_mov([f for f in mov["resto"] if f[3]], "Resto (solo días de señal)"))
    L.append("")
    L.append("== 2) SEÑALES Y ¿CUÁNTAS LLEGAN A +1R Y +2R EN 20 RUEDAS? ==")
    for g, nom in (("def", "Defensivas"), ("resto", "Resto")):
        a = AL[AL["g"] == g]
        if not len(a): continue
        acc = a["s"].value_counts()
        L.append(f"{nom:<11}: {len(a):>5} señales en {acc.size:>3} acciones{'  ← MENOS DE 200: POCO CONFIABLE' if len(a) < MIN_OPS else ''}. ATR % mediano {100 * a['atrp'].median():.1f}%, distancia mediana al stop {100 * a['riesgo'].median():.1f}%, "
                 f"llegan a +1R {100 * a['a1'].mean():.0f}%, a +2R {100 * a['a2'].mean():.0f}%. Más señales: {', '.join(f'{k} ({v})' for k, v in acc.head(8).items())}.")
    a = AL[AL["g"] == "def"]
    if len(a): L.append(f"De las señales de defensivas, el {100 * (a['atrp'] < 0.02).mean():.0f}% cae en el grupo de ATR % menor a 2%; del resto, el {100 * (AL[AL['g'] == 'resto']['atrp'] < 0.02).mean():.0f}%.")
    L.append("")
    enc = f"{'Versión':<46}| {'ops':>5} | {'toma.':>5} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'dur.':>4} | {'acum.':>8} | {'c. MTM':>7} | {'efec./mes':>9} | {'mes+':>4} | {'peor mes':>8}"
    for g, nom in (("def", "DEFENSIVAS"), ("resto", "RESTO")):
        L += [f"== 3) RESULTADOS: {nom} ==", enc]
        res = {}
        for clave, nombre, *_ in VERS:
            sub = DF[clave][DF[clave]["g"] == g]; x = V.metricas(sub, dias); res[clave] = (sub, x)
            L.append(V.linea(nombre, x, "*" if x and x["ops"] < MIN_OPS else ""))
        sub = res["D20"][0]; gs = sub[sub["siguio"]] if len(sub) else sub
        if len(gs):
            post = np.array([p[-1] - p[19] for p in gs["path"]])
            L.append(f"   D20: siguieron {len(gs)} de {len(sub)} ({100 * len(gs) / len(sub):.0f}%); después de la 20 ganaron en promedio {100 * post.mean():+.2f}% (mediana {100 * np.median(post):+.2f}%), mejoraron {100 * (post > 0).mean():.0f}%.")
        L.append(f"-- por mitades (señales hasta {corte.date()} | desde {corte.date()}; el capital vuelve a empezar en cada mitad) --")
        rk1, rk2 = [], []
        for clave in ("A20", "D20", "E", "A30", "A40"):
            sub = res[clave][0]; a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
            xa, xb = V.metricas(a, d1), V.metricas(b, d2)
            fa = (f"{len(a):>4}{'*' if len(a) < MIN_OPS else ' '} ops, acum. {100 * xa['acu']:+7.1f}%, R {xa['R']:+5.2f}, c. MTM {100 * xa['mtm']:+6.1f}%") if xa else "sin operaciones"
            fb = (f"{len(b):>4}{'*' if len(b) < MIN_OPS else ' '} ops, acum. {100 * xb['acu']:+7.1f}%, R {xb['R']:+5.2f}, c. MTM {100 * xb['mtm']:+6.1f}%") if xb else "sin operaciones"
            L.append(f"   {clave:<4}| 1.ª: {fa} || 2.ª: {fb}")
            if xa: rk1.append((clave, xa["acu"]))
            if xb: rk2.append((clave, xb["acu"]))
        L.append(f"   Orden por acumulado: 1.ª mitad {' > '.join(k for k, _ in sorted(rk1, key=lambda t: -t[1]))} | 2.ª mitad {' > '.join(k for k, _ in sorted(rk2, key=lambda t: -t[1]))}")
        L.append("")
    tot, dd, _ = F.spy_bh(ini, ult, ini_dl)
    L.append(f"Referencia: SPY comprar y mantener en el mismo período: {100 * tot:+.1f}% (caída máxima {100 * dd:.1f}%).")
    L.append("(* = grupo o mitad con menos de 200 operaciones: poco confiable.)")
    txt = "\n".join(L); open("data/investigacion_defensivas.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
