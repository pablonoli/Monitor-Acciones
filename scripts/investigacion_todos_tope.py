"""Investigación: con TODOS los CEDEARs, ¿un tope de posiciones abiertas mejora la relación entre resultado y caída? 5 años, plan A (cierra todo en la rueda 20).
Plan A: entrada en la apertura siguiente, costo 0,15% por lado, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra, cierre al final de la rueda 20.
Capital: 1% de riesgo por operación, exposición máx. 100%. Con tope de N posiciones abiertas, si ya hay N abiertas la señal se saltea; entre las señales del mismo día entra primero la de mayor Z.
Conjuntos: TODAS las acciones y las 122 de antes (referencia). Filtros: ATR 2% o más; Strong + ATR 2% o más. Topes: sin tope, 8, 10, 12 y 15.
Resultado: data/investigacion_todos_tope.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F
import investigacion_circulos_20_vol as V
import investigacion_circulos_20_sel as Sx

CHICA = 0.15
TOPES = [None, 15, 12, 10, 8]


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
    conj = [("TODAS", S), ("Las 122 de antes", S[~S["nuevo"]])]
    filt = [("ATR 2% o más", lambda x: x[x["atrp"] >= 0.02]), ("Strong + ATR 2% o más", lambda x: x[x["strong"] & (x["atrp"] >= 0.02)])]
    L = [f"TOPE DE POSICIONES ABIERTAS CON TODOS LOS CEDEARS. Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones bajadas, {int((~pd.Series(sorted(datos)).isin(orig)).sum())} nuevas contra las 122 de antes.",
         "Plan A (cierra todo en la rueda 20): entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra. Costo 0,15% por lado. 1% de riesgo por operación, exposición máx. 100%.",
         "Con tope de N abiertas, si ya hay N posiciones abiertas la señal se saltea. Entre las señales del mismo día entra primero la de mayor Z.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%. Se probaron {len(conj) * len(filt) * len(TOPES)} combinaciones (sesgo de selección: la mejor suele rendir menos hacia adelante). Diferencia menor a 15 pp = puede ser casualidad.",
         "'acum./caída' = puntos de resultado por cada punto de caída máxima (valuando las abiertas a mercado): cuanto más alto, mejor relación.",
         "LÍMITES: lista de CEDEARs de hoy (sesgo de supervivencia); las operaciones se superponen y no son independientes.", ""]
    enc = (f"{'Conjunto | filtro | tope':<54}| {'señales':>7} | {'toma.':>5} | {'por mes':>7} | {'abiertas':>8} | {'prom.':>7} | {'gan.':>4} | {'R med':>5} | {'acum.':>8} | {'c. MTM':>7} | {'acum./caída':>11} | {'mes+':>4} | {'peor mes':>8} | {'1.ª mitad':>9} | {'2.ª mitad':>9}")
    L += ["== RESULTADOS (5 años) ==", enc]
    for cn, cs in conj:
        for fn, ff in filt:
            sub = ff(cs)
            for tp in TOPES:
                x = Sx.met(sub, dias, tp, "z") if len(sub) else None
                a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
                xa = Sx.met(a, d1, tp, "z") if len(a) else None; xb = Sx.met(b, d2, tp, "z") if len(b) else None
                nom = f"{cn} | {fn} | {'sin tope' if tp is None else 'máx. ' + str(tp)}"
                if x is None: L.append(f"{nom:<54}| sin operaciones"); continue
                ha = f"{100 * xa['acu']:+8.1f}%" if xa else f"{'—':>9}"; hb = f"{100 * xb['acu']:+8.1f}%" if xb else f"{'—':>9}"
                rel = x["acu"] / abs(x["mtm"]) if x["mtm"] else float("nan")
                L.append(f"{nom:<54}| {x['sen']:>7} | {x['tom']:>5} | {x['pmes']:>7.1f} | {x['abiertas']:>8.1f} | {100 * x['prom']:+6.2f}% | {100 * x['gan']:3.0f}% | {x['R']:+5.2f} | {100 * x['acu']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {rel:>11.1f} | {100 * x['mesp']:3.0f}% | {100 * x['peor']:+7.1f}% | {ha} | {hb}")
            L.append("")
    txt = "\n".join(L); open("data/investigacion_todos_tope.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
