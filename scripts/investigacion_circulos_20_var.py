"""Investigación: ¿conviene otro tope de días u otro umbral para seguir después de la rueda 20 con los círculos verdes? 5 años, CEDEARs, sin filtro.
Mismas convenciones que investigacion_circulos_20.py y investigacion_circulos_20_vol.py: entrada en la apertura siguiente, costo 0,15% por lado,
stop inicial = mínimo de 20 ruedas, stop antes que objetivo en la misma vela, gaps al open, vende la mitad en el objetivo y mueve el stop al precio de compra;
capital con 1% de riesgo por operación, exposición máx. 100%.
Variantes: A y D de referencia; topes fijos 30/40/50; D con umbrales +0,5R/+1,5R/+2R; D con tope 30/50; vender la mitad a +1R o +1,5R (con A y con D); E (máx. 60) como referencia.
Resultado: data/investigacion_circulos_20_var.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F
import investigacion_circulos_20_vol as V

K = F.K
DEC = 20
GRUPOS = V.GRUPOS
MIN_OPS = V.MIN_OPS
CHICA = 0.15   # diferencia de acumulado contra A por debajo de la cual puede ser casualidad

# (clave, nombre, tope sin regla, umbral en R para seguir (None = sin regla), tope si sigue, objetivo de la mitad en R)
VARS = [("A", "A. Cierra todo en la 20 (mitad a +2R)", 20, None, 20, 2.0),
        ("D", "D. En la 20, si +1R sigue hasta la 40", 20, 1.0, 40, 2.0),
        ("T30", "1. Cierra todo en la 30", 30, None, 30, 2.0),
        ("T40", "1. Cierra todo en la 40", 40, None, 40, 2.0),
        ("T50", "1. Cierra todo en la 50", 50, None, 50, 2.0),
        ("D05", "2. En la 20, si +0,5R sigue hasta la 40", 20, 0.5, 40, 2.0),
        ("D15", "2. En la 20, si +1,5R sigue hasta la 40", 20, 1.5, 40, 2.0),
        ("D20", "2. En la 20, si +2R sigue hasta la 40", 20, 2.0, 40, 2.0),
        ("D1-30", "3. En la 20, si +1R sigue hasta la 30", 20, 1.0, 30, 2.0),
        ("D1-50", "3. En la 20, si +1R sigue hasta la 50", 20, 1.0, 50, 2.0),
        ("A-m1", "4. A con mitad a +1R", 20, None, 20, 1.0),
        ("D-m1", "4. D con mitad a +1R", 20, 1.0, 40, 1.0),
        ("A-m15", "5. A con mitad a +1,5R", 20, None, 20, 1.5),
        ("D-m15", "5. D con mitad a +1,5R", 20, 1.0, 40, 1.5),
        ("E", "Ref. E: sin límite (máx. 60)", 60, None, 60, 2.0)]
UMBRALES = [0.5, 1.0, 1.5, 2.0]


def camino(P, e, stop, tope, umbral, tope_sigue, objm):
    ent = P["O"][e]; R = ent - stop; obj = ent + objm * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []; hk = None
    fin = e + tope; siguio = False; j = e
    while j < min(n, fin):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if lo <= sc:
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path, ((hk, vm) if mitad else None), siguio
        if not mitad and hi >= obj:
            vm = max(obj, o) / ent * K; mitad = True; sc = ent; hk = j - e
        path.append(0.5 * vm + 0.5 * (c / ent * K) if mitad else c / ent * K)
        if umbral is not None and j - e + 1 == DEC:
            if c >= ent + umbral * R:
                siguio = True; fin = e + tope_sigue; sc = max(sc, ent)
            else:
                return path, ((hk, vm) if mitad else None), siguio
        j += 1
    return path, ((hk, vm) if mitad else None), siguio


def alcanza(P, e, stop, ruedas=DEC):
    """Máximo alcanzado (en R) dentro de las primeras `ruedas` ruedas antes de tocar el stop inicial (stop primero en la misma vela)."""
    ent = P["O"][e]; R = ent - stop; n = len(P["C"]); mx = -9.0
    for j in range(e, min(n, e + ruedas)):
        if P["L"][j] <= stop: break
        mx = max(mx, (P["H"][j] - ent) / R)
    return mx


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    sb_acu, sb_dd, _ = F.spy_bh(ini, ult, ini_dl)
    filas = {v[0]: [] for v in VARS}; alc = []
    for s, P in datos.items():
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            riesgo = (ent - stop) / ent; atrp = float(P["atrp"][i])
            alc.append(dict(atrp=atrp, mx=alcanza(P, e, stop)))
            for clave, _, tope, umbral, tope_sigue, objm in VARS:
                path, half, siguio = camino(P, e, stop, tope, umbral, tope_sigue, objm); ret = path[-1] - 1
                filas[clave].append(dict(s=s, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0,
                                         path=path, half=half, dts=list(P["f"][e:e + len(path)]), dur=len(path), siguio=siguio, atrp=atrp))
    DF = {k: pd.DataFrame(v) for k, v in filas.items()}; AL = pd.DataFrame(alc)
    probadas = len(VARS) - 1   # sin contar E (referencia ya vista)
    L = [f"¿OTRO TOPE DE DÍAS U OTRO UMBRAL PARA LOS CÍRCULOS VERDES? Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Sin filtro.",
         f"Plan base: entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, vende la mitad en el objetivo y mueve el stop al precio de compra. Costo {100 * I.COSTO:.2f}% por lado. Stop antes que objetivo; gaps al open.",
         "Capital: 1% de riesgo por operación, sin apalancamiento (exposición máx. 100%). La rueda 1 es la de entrada. 'Si +XR sigue hasta la N' = al cierre de la rueda 20, si está en precio de compra + X×R o más, sigue con stop en el precio de compra hasta la rueda N; si no, cierra.",
         f"VARIANTES PROBADAS EN ESTA CORRIDA: {probadas} (más E como referencia). Contando las pruebas anteriores de la regla de salida (9 planes + 5 versiones A–E + 7 por volatilidad), ya van más de 35 combinaciones.",
         f"Marca '≈A' = diferencia de acumulado contra A menor a {int(100 * CHICA)} puntos: puede ser casualidad.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%.",
         "LÍMITES: sesgo de selección (con tantas variantes, la mejor casi siempre rinde menos en el futuro de lo que muestra acá); sesgo de supervivencia (lista de CEDEARs de hoy);",
         "las operaciones se superponen y no son independientes; los grupos de volatilidad tienen menos operaciones que el total.", ""]
    L.append("== ¿QUÉ % DE LAS OPERACIONES LLEGA A CADA UMBRAL DENTRO DE LAS PRIMERAS 20 RUEDAS (antes de tocar el stop inicial)? ==")
    tit = f"{'Grupo':<24}| " + " | ".join(f"+{str(u).replace('.', ',')}R" for u in UMBRALES) + " | ops"
    L.append(tit)
    for g, lo, hi in [("Todas", 0.0, 9.0)] + GRUPOS:
        a = AL[(AL["atrp"] >= lo) & (AL["atrp"] < hi)]
        L.append(f"{g:<24}| " + " | ".join(f"{100 * (a['mx'] >= u).mean():4.0f}%" for u in UMBRALES) + f" | {len(a)}")
    L.append("")
    enc = f"{'Variante':<46}| {'ops':>5} | {'toma.':>5} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'dur.':>4} | {'acum.':>8} | {'c. MTM':>7} | {'efec./mes':>9} | {'mes+':>4} | {'peor mes':>8} | vs A"
    L += ["== TODAS LAS ACCIONES ==", enc]
    X = {}
    for clave, nombre, *_ in VARS:
        X[clave] = V.metricas(DF[clave], dias)
    aA = X["A"]["acu"]
    for clave, nombre, *_ in VARS:
        x = X[clave]; dif = x["acu"] - aA
        nota = "" if clave == "A" else f"{100 * dif:+6.1f} pp{'  ≈A' if abs(dif) < CHICA else ''}"
        L.append(V.linea(nombre, x) + f" | {nota}")
    L.append("")
    # después de la 20 para las variantes con regla
    L.append("== OPERACIONES QUE SIGUIERON DESPUÉS DE LA RUEDA 20 (variantes con regla) ==")
    for clave, nombre, tope, umbral, *_ in VARS:
        if umbral is None: continue
        sub = DF[clave]; gs = sub[sub["siguio"]]
        if not len(gs): L.append(f"  {clave}: ninguna"); continue
        post = np.array([p[-1] - p[DEC - 1] for p in gs["path"]])
        L.append(f"  {clave:<6}: siguieron {len(gs)} de {len(sub)} ({100 * len(gs) / len(sub):.0f}%); después de la 20: promedio {100 * post.mean():+.2f}%, mediana {100 * np.median(post):+.2f}%, mejoraron {100 * (post > 0).mean():.0f}%.")
    L.append("")
    # mitades
    L += [f"== POR MITADES (señales hasta {corte.date()} | desde {corte.date()}; el capital vuelve a empezar en cada mitad) ==",
          f"{'Variante':<46}| {'1.ª acum.':>9} | {'R med':>5} | {'c. MTM':>7} | {'mes+':>4} || {'2.ª acum.':>9} | {'R med':>5} | {'c. MTM':>7} | {'mes+':>4}"]
    M1, M2 = {}, {}
    for clave, nombre, *_ in VARS:
        sub = DF[clave]; a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
        xa, xb = V.metricas(a, d1), V.metricas(b, d2); M1[clave], M2[clave] = xa["acu"], xb["acu"]
        L.append(f"{nombre:<46}| {100 * xa['acu']:+8.1f}% | {xa['R']:+5.2f} | {100 * xa['mtm']:+6.1f}% | {100 * xa['mesp']:3.0f}% || {100 * xb['acu']:+8.1f}% | {xb['R']:+5.2f} | {100 * xb['mtm']:+6.1f}% | {100 * xb['mesp']:3.0f}%")
    r1 = [k for k, _ in sorted(M1.items(), key=lambda t: -t[1])]; r2 = [k for k, _ in sorted(M2.items(), key=lambda t: -t[1])]
    L.append(f"  Puesto en la 1.ª mitad: {', '.join(f'{k} {r1.index(k) + 1}.º' for k, *_ in VARS)}")
    L.append(f"  Puesto en la 2.ª mitad: {', '.join(f'{k} {r2.index(k) + 1}.º' for k, *_ in VARS)}")
    sin_e = [k for k, *_ in VARS if k != "E"]
    top3 = [k for k, _ in sorted(((k, X[k]["acu"]) for k in sin_e), key=lambda t: -t[1])][:3]
    gana_ambas = [k for k in sin_e if k != "A" and M1[k] > M1["A"] and M2[k] > M2["A"]]
    L.append(f"  Mejores 3 en 5 años (sin contar E): {', '.join(top3)}. Le ganan a A en las DOS mitades: {', '.join(gana_ambas) or 'ninguna'}.")
    L.append("")
    # por volatilidad, solo A + top 3
    L.append(f"== POR VOLATILIDAD (A y las 3 mejores: {', '.join(top3)}). Cada grupo con su propio capital. ==")
    for g, lo, hi in GRUPOS:
        L += [f"-- {g}", enc]
        xg = {}
        for clave in ["A"] + [k for k in top3 if k != "A"]:
            nombre = next(v[1] for v in VARS if v[0] == clave)
            sub = DF[clave]; sub = sub[(sub["atrp"] >= lo) & (sub["atrp"] < hi)]
            xg[clave] = V.metricas(sub, dias)
        for clave, x in xg.items():
            nombre = next(v[1] for v in VARS if v[0] == clave)
            if x is None or xg["A"] is None: L.append(V.linea(nombre, x)); continue
            dif = x["acu"] - xg["A"]["acu"]
            nota = "" if clave == "A" else f"{100 * dif:+6.1f} pp{'  ≈A' if abs(dif) < CHICA else ''}"
            L.append(V.linea(nombre, x, "*" if x["ops"] < MIN_OPS else "") + f" | {nota}")
    L.append("")
    L.append("(* = menos de 200 operaciones: poco confiable.)")
    txt = "\n".join(L); open("data/investigacion_circulos_20_var.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
