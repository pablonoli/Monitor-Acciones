"""Investigación: la estrategia del círculo verde (plan A, cierra todo en la rueda 20) con TODOS los CEDEARs contra las 122 acciones de antes y contra solo las nuevas. 5 años.
Plan A: entrada en la apertura siguiente, costo 0,15% por lado, stop inicial = mínimo de 20 ruedas, stop antes que objetivo en la misma vela, gaps al open, mitad a +2R y stop al precio de compra, cierre al final de la rueda 20.
Capital: 1% de riesgo por operación (tamaño = 1% / distancia al stop), sin apalancamiento (exposición máx. 100%: si no hay lugar, se saltea).
Cuando hay más señales que lugar, el orden de elección importa: se muestran dos criterios, "alfabético" (el de las pruebas anteriores, arbitrario) y "mayor Z" (la más fuerte primero).
Resultado: data/investigacion_todos.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F
import investigacion_circulos_20_vol as V
import investigacion_circulos_20_sel as Sx

CHICA = 0.15


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys()); orig = set(json.load(open("data/cedears_originales.json")))
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    crudos = I.bajar(simbolos, ini_dl); datos = {s: I.preparar(x) for s, x in crudos.items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    sb_acu, sb_dd, _ = F.spy_bh(ini, ult, ini_dl)
    filas = []
    for s, P in datos.items():
        c = crudos[s]["Close"]; mom = c.diff(); Z = ((mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)).values; n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            path, half, _ = V.camino(P, e, stop, 20, None, 20, 20); ret = path[-1] - 1; riesgo = (ent - stop) / ent
            filas.append(dict(s=s, nuevo=s not in orig, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0, path=path, half=half,
                              dts=list(P["f"][e:e + len(path)]), dur=len(path), atrp=float(P["atrp"][i]), strong=bool(P["strong"][i]), z=float(Z[i])))
    S = pd.DataFrame(filas)
    conj = [("Las 122 de antes", S[~S["nuevo"]]), ("TODAS (nuevas + antes)", S), ("Solo las nuevas", S[S["nuevo"]])]
    filt = [("todas las señales", lambda x: x), ("ATR 2% o más", lambda x: x[x["atrp"] >= 0.02]), ("solo Strong", lambda x: x[x["strong"]]), ("Strong + ATR 2% o más", lambda x: x[x["strong"] & (x["atrp"] >= 0.02)])]
    meses = len(pd.period_range(dias[0], dias[-1], freq="M"))
    L = [f"LA ESTRATEGIA CON TODOS LOS CEDEARS. Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones bajadas, {int((~pd.Series(sorted(datos)).isin(orig)).sum())} nuevas contra las 122 de antes.",
         "Plan A (cierra todo en la rueda 20): entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra. Costo 0,15% por lado. 1% de riesgo por operación, exposición máx. 100% (si no hay plata libre, la señal se saltea).",
         "Orden de elección cuando hay más señales que lugar: 'alfab.' = alfabético (como en las pruebas anteriores, arbitrario); 'Z' = la de mayor Z primero.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%. Se probaron {len(conj) * len(filt) * 2} combinaciones: la mejor suele rendir menos hacia adelante (sesgo de selección). Diferencia menor a 15 pp = puede ser casualidad.",
         "LÍMITES: lista de CEDEARs de hoy (sesgo de supervivencia, más fuerte en las nuevas: muchas son acciones o ETFs que subieron mucho); las operaciones se superponen y no son independientes. Z y ATR % se miden el día de la señal.", ""]
    enc = (f"{'Conjunto | filtro | orden':<62}| {'señales':>7} | {'toma.':>5} | {'por mes':>7} | {'abiertas':>8} | {'prom.':>7} | {'gan.':>4} | {'R med':>5} | {'acum.':>8} | {'c. MTM':>7} | {'mes+':>4} | {'peor mes':>8} | {'1.ª mitad':>9} | {'2.ª mitad':>9}")
    L += ["== RESULTADOS (5 años) ==", enc]
    res = {}
    for cn, cs in conj:
        for fn, ff in filt:
            sub = ff(cs)
            for on, pr in (("alfab.", None), ("Z", "z")):
                x = Sx.met(sub, dias, None, pr) if len(sub) else None
                a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
                xa = Sx.met(a, d1, None, pr) if len(a) else None; xb = Sx.met(b, d2, None, pr) if len(b) else None
                res[(cn, fn, on)] = x
                nom = f"{cn} | {fn} | {on}"
                if x is None: L.append(f"{nom:<62}| sin operaciones"); continue
                ha = f"{100 * xa['acu']:+8.1f}%" if xa else f"{'—':>9}"; hb = f"{100 * xb['acu']:+8.1f}%" if xb else f"{'—':>9}"
                L.append(f"{nom:<62}| {x['sen']:>7} | {x['tom']:>5} | {x['pmes']:>7.1f} | {x['abiertas']:>8.1f} | {100 * x['prom']:+6.2f}% | {100 * x['gan']:3.0f}% | {x['R']:+5.2f} | {100 * x['acum']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {100 * x['mesp']:3.0f}% | {100 * x['peor']:+7.1f}% | {ha} | {hb}")
        L.append("")
    L.append("== LAS SEÑALES NUEVAS: ¿CUÁNTAS HAY Y QUÉ ACCIONES MÁS APARECEN? ==")
    nv = S[S["nuevo"]]; L.append(f"Señales de las acciones nuevas: {len(nv)} (contra {int((~S['nuevo']).sum())} de las 122 de antes). Con ATR 2% o más: {int((nv['atrp'] >= 0.02).sum())}. Strong con ATR 2% o más: {int((nv['strong'] & (nv['atrp'] >= 0.02)).sum())}.")
    top = nv["s"].value_counts().head(15); L.append("Más señales entre las nuevas: " + ", ".join(f"{k} ({v})" for k, v in top.items()))
    L.append("")
    L.append("== LAS NUEVAS CON MEJOR Y PEOR RESULTADO (ATR 2% o más; operaciones sueltas, sin capital; mínimo 8 señales) ==")
    g = nv[nv["atrp"] >= 0.02].groupby("s").agg(n=("rnet", "size"), R=("rnet", "mean"), gan=("ret", lambda r: (r > 0).mean()), prom=("ret", "mean")).query("n >= 8").sort_values("R", ascending=False)
    for k, r in pd.concat([g.head(8), g.tail(8)]).iterrows(): L.append(f"  {k:<6} {int(r['n']):>3} señales, R medio {r['R']:+5.2f}, ganadoras {100 * r['gan']:3.0f}%, promedio {100 * r['prom']:+6.2f}%")
    L.append("(Mejor y peor por acción son resultados con pocas señales: no sirven para elegir acciones, solo para ver dispersión.)")
    txt = "\n".join(L); open("data/investigacion_todos.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
