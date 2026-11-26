"""Investigación: ¿cuánto resultado se pierde (o se gana) si se toman menos señales de círculo verde? 5 años, CEDEARs, plan A.
Plan A (mismas convenciones que investigacion_circulos_20.py): entrada en la apertura siguiente, costo 0,15% por lado, stop inicial = mínimo de 20 ruedas,
stop antes que objetivo en la misma vela, gaps al open, vende la mitad a +2R y mueve el stop al precio de compra, cierra todo al cierre de la rueda 20.
Capital: 1% de riesgo por operación (tamaño = 1% / distancia al stop), sin apalancamiento (exposición máx. 100%: si no hay lugar, se saltea).
Versiones: todas; solo Strong; solo volatilidad media; medias + altas; tope de 5/10/15 posiciones abiertas (prioridad mayor Z o menor ATR %);
máximo 2 señales nuevas por semana (mayor Z); combinación de la mejor con Strong. Resultado: data/investigacion_circulos_20_sel.txt"""
import json, sys, heapq
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F
import investigacion_circulos_20_vol as V

CHICA = 0.15


def sim_sel(df, maxpos=None, prio=None, riesgo_op=0.01, tope=1.0):
    """Como investigacion_circulos_filtros.sim, pero con tope de posiciones abiertas a la vez y prioridad entre las señales del mismo día de entrada.
    prio: "z" (mayor Z primero), "atr" (menor ATR % primero) o None (orden alfabético, como sim)."""
    if not len(df): return None
    df = df.copy()
    if prio == "z": df["_k"] = -df["z"]
    elif prio == "atr": df["_k"] = df["atrp"]
    else: df["_k"] = 0.0
    df = df.sort_values(["fe", "_k", "s"])
    eq, expo, ab = 1.0, 0.0, []; eventos, tomadas, filas = [], [], []; pico, dd, n = 1.0, 0.0, 0
    def cerrar(hasta):
        nonlocal eq, expo, pico, dd
        while ab and (hasta is None or ab[0][0] <= hasta):
            fx, _, tam, pnl = heapq.heappop(ab); eq += pnl; expo -= tam; pico = max(pico, eq); dd = min(dd, eq / pico - 1); eventos.append((fx, eq))
    for r in df.itertuples():
        cerrar(r.fe)
        if maxpos is not None and len(ab) >= maxpos: continue
        tam = min(riesgo_op / r.riesgo, tope - expo)
        if tam < 0.02: continue
        n += 1; expo += tam; heapq.heappush(ab, (r.fx, n, tam, eq * tam * r.ret))
        tomadas.append((eq * tam, tam, r.path, r.dts, r.half)); filas.append(r.Index)
    cerrar(None)
    por, prev = {}, 1.0
    for a in sorted({e[0].year for e in eventos}):
        fin = [e[1] for e in eventos if e[0].year == a][-1]; por[a] = fin / prev - 1; prev = fin
    return dict(tom=n, acu=eq - 1, dd=dd, por=por, eventos=eventos, tomadas=tomadas, filas=df.loc[filas])


def met(df, dias, maxpos=None, prio=None):
    res = sim_sel(df, maxpos, prio)
    if res is None or res["tom"] == 0: return None
    c = F.caja_datos(res, dias); mm = F.mercado(res, dias); t = res["filas"]
    r, R = t["ret"].values, t["rnet"].values; meses = max(1, len(pd.period_range(dias[0], dias[-1], freq="M")))
    return dict(res=res, sen=len(df), tom=res["tom"], pmes=res["tom"] / meses, abiertas=mm[1], prom=r.mean(), med=np.median(r), gan=(r > 0).mean(), R=R.mean(),
                acu=res["acu"], mtm=mm[0], efec=c["inflow"].mean(), mesp=(c["pnl"] > 0).mean(), peor=c["pnl"].min(), anio=t["fe"].dt.year.value_counts().sort_index())


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    crudos = I.bajar(simbolos, ini_dl); datos = {s: I.preparar(x) for s, x in crudos.items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    sb_acu, sb_dd, sb_por = F.spy_bh(ini, ult, ini_dl)
    filas = []
    for s, P in datos.items():
        c = crudos[s]["Close"]; mom = c.diff(); Z = ((mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)).values
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            path, half, _ = V.camino(P, e, stop, 20, None, 20, 20); ret = path[-1] - 1; riesgo = (ent - stop) / ent
            filas.append(dict(s=s, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0, path=path, half=half,
                              dts=list(P["f"][e:e + len(path)]), dur=len(path), atrp=float(P["atrp"][i]), strong=bool(P["strong"][i]), z=float(Z[i])))
    S = pd.DataFrame(filas)
    sem = S.assign(_sem=S["fe"].dt.to_period("W")).sort_values(["_sem", "z"], ascending=[True, False])
    dos_sem = S.loc[sem.groupby("_sem").head(2).index]
    # (clave, nombre, subconjunto, tope de posiciones, prioridad)
    VERS = [("1", "1. Todas las señales (base)", S, None, None),
            ("2", "2. Solo Strong", S[S["strong"]], None, None),
            ("3", "3. Solo volatilidad media (ATR 2% a 4%)", S[(S["atrp"] >= 0.02) & (S["atrp"] < 0.04)], None, None),
            ("4", "4. Medias y altas (ATR > 2%)", S[S["atrp"] >= 0.02], None, None)]
    for mp in (5, 10, 15):
        VERS.append((f"5-{mp}z", f"5. Máx. {mp} abiertas, prioridad mayor Z", S, mp, "z"))
        VERS.append((f"5-{mp}a", f"5. Máx. {mp} abiertas, prioridad menor ATR %", S, mp, "atr"))
    VERS.append(("6", "6. Máx. 2 señales nuevas por semana (mayor Z)", dos_sem, None, "z"))
    X = {k: met(sub, dias, mp, pr) for k, _, sub, mp, pr in VERS}
    # 7: la mejor de 2..6 (por acumulado) combinada con Strong
    cand = [k for k, *_ in VERS if k != "1" and X[k]]
    mejor = max(cand, key=lambda k: X[k]["acu"]); vm = next(v for v in VERS if v[0] == mejor)
    sub7 = vm[2][vm[2]["strong"]] if mejor != "2" else S[S["strong"]]
    if mejor == "6": sub7 = dos_sem[dos_sem["strong"]]
    nombre7 = f"7. La mejor ({mejor}) + solo Strong"
    VERS.append(("7", nombre7, sub7, vm[3], vm[4])); X["7"] = met(sub7, dias, vm[3], vm[4])
    total = len(VERS)
    meses = len(pd.period_range(dias[0], dias[-1], freq="M"))
    L = [f"¿CUÁNTO SE PIERDE O SE GANA TOMANDO MENOS SEÑALES? Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Plan A (cierra todo en la rueda 20).",
         f"Plan A: entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra, cierre al final de la rueda 20. Costo {100 * I.COSTO:.2f}% por lado.",
         "Capital: 1% de riesgo por operación, sin apalancamiento (exposición máx. 100%: si no hay plata libre, la señal se saltea, aunque esté elegida).",
         f"IMPORTANTE: aun 'tomando todas', el capital no alcanza para todas: de {len(S)} señales ({len(S) / meses:.0f} por mes) se toman las que entran en el 100% de exposición.",
         "Por eso las estadísticas por operación (prom., med., gan., R) se calculan sobre las operaciones realmente tomadas.",
         f"VERSIONES PROBADAS EN ESTA CORRIDA: {total}. Marca '≈base' = diferencia de acumulado contra la versión 1 menor a {int(100 * CHICA)} puntos: puede ser casualidad.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%.",
         "LÍMITES: sesgo de selección (se elige la mejor entre varias); sesgo de supervivencia (lista de CEDEARs de hoy); las operaciones se superponen y no son independientes;",
         "Z y ATR % se miden el día de la señal (se conocen antes de comprar, no hay trampa de mirar el futuro).", ""]
    enc = (f"{'Versión':<50}| {'señales':>7} | {'toma.':>5} | {'por mes':>7} | {'abiertas':>8} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'acum.':>8} | {'c. MTM':>7} | "
           f"{'efec./mes':>9} | {'mes+':>4} | {'peor mes':>8} | vs base")
    L += ["== RESULTADOS (5 años) ==", enc]
    base = X["1"]["acu"]
    for k, nombre, *_ in VERS:
        x = X[k]
        if x is None: L.append(f"{nombre:<50}| sin operaciones"); continue
        dif = x["acu"] - base; nota = "" if k == "1" else f"{100 * dif:+6.1f} pp{'  ≈base' if abs(dif) < CHICA else ''}"
        L.append(f"{nombre:<50}| {x['sen']:>7} | {x['tom']:>5} | {x['pmes']:>7.1f} | {x['abiertas']:>8.1f} | {100 * x['prom']:+6.2f}% | {100 * x['med']:+6.2f}% | {100 * x['gan']:3.0f}% | {x['R']:+5.2f} | "
                 f"{100 * x['acu']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {100 * x['efec']:8.0f}% | {100 * x['mesp']:3.0f}% | {100 * x['peor']:+7.1f}% | {nota}")
    L.append("abiertas = promedio de posiciones abiertas a la vez; por mes = operaciones tomadas por mes; efec./mes = capital que vuelve a efectivo por mes.")
    L.append("")
    anios_l = sorted({a for x in X.values() if x for a in x["res"]["por"]} | set(sb_por))
    L += ["== OPERACIONES TOMADAS POR AÑO (por año de entrada) ==", f"{'Versión':<50}" + "".join(f"{a:>8}" for a in anios_l)]
    for k, nombre, *_ in VERS:
        x = X[k]
        if x: L.append(f"{nombre:<50}" + "".join(f"{int(x['anio'].get(a, 0)):>8}" for a in anios_l))
    L += ["", "== RETORNO DEL CAPITAL POR AÑO (por año de cierre de las operaciones; SPY: año calendario) ==", f"{'Versión':<50}" + "".join(f"{a:>10}" for a in anios_l)]
    for k, nombre, *_ in VERS:
        x = X[k]
        if x: L.append(f"{nombre:<50}" + "".join(f"{100 * x['res']['por'][a]:+9.1f}%" if a in x["res"]["por"] else f"{'—':>10}" for a in anios_l))
    L.append(f"{'SPY comprar y mantener':<50}" + "".join(f"{100 * sb_por[a]:+9.1f}%" if a in sb_por else f"{'—':>10}" for a in anios_l))
    L += ["", f"== POR MITADES (señales hasta {corte.date()} | desde {corte.date()}; el capital vuelve a empezar en cada mitad) ==",
          f"{'Versión':<50}| {'1.ª acum.':>9} | {'por mes':>7} | {'R med':>5} | {'c. MTM':>7} || {'2.ª acum.':>9} | {'por mes':>7} | {'R med':>5} | {'c. MTM':>7}"]
    M1, M2 = {}, {}
    for k, nombre, sub, mp, pr in VERS:
        a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
        xa, xb = met(a, d1, mp, pr), met(b, d2, mp, pr)
        if not xa or not xb: L.append(f"{nombre:<50}| sin operaciones en alguna mitad"); continue
        M1[k], M2[k] = xa["acu"], xb["acu"]
        L.append(f"{nombre:<50}| {100 * xa['acu']:+8.1f}% | {xa['pmes']:>7.1f} | {xa['R']:+5.2f} | {100 * xa['mtm']:+6.1f}% || {100 * xb['acu']:+8.1f}% | {xb['pmes']:>7.1f} | {xb['R']:+5.2f} | {100 * xb['mtm']:+6.1f}%")
    r1 = sorted(M1, key=lambda k: -M1[k]); r2 = sorted(M2, key=lambda k: -M2[k])
    L.append(f"  Puesto 1.ª mitad: {', '.join(f'{k} {r1.index(k) + 1}.º' for k in M1)}")
    L.append(f"  Puesto 2.ª mitad: {', '.join(f'{k} {r2.index(k) + 1}.º' for k in M2)}")
    ambas = [k for k in M1 if k != "1" and k in M2 and M1[k] > M1["1"] and M2[k] > M2["1"]]
    L.append(f"  Le ganan a la base (1) en las DOS mitades: {', '.join(ambas) or 'ninguna'}.")
    L += ["", "== OPCIONES CON 5 A 10 OPERACIONES POR MES ==" ]
    sel = [(k, n) for k, n, *_ in VERS if X[k] and 5 <= X[k]["pmes"] <= 10]
    for k, n in sorted(sel, key=lambda t: -X[t[0]]["acu"]):
        x = X[k]; L.append(f"  {n:<50}: {x['pmes']:.1f} por mes, acumulado {100 * x['acu']:+.1f}% ({100 * (x['acu'] - base):+.1f} pp contra la base), caída MTM {100 * x['mtm']:+.1f}%, abiertas {x['abiertas']:.1f}")
    if not sel: L.append("  ninguna versión quedó entre 5 y 10 operaciones por mes")
    txt = "\n".join(L); open("data/investigacion_circulos_20_sel.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
