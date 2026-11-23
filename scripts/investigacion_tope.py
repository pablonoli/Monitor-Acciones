"""Investigación (no es parte del dashboard): ¿conviene NO comprar por tendencia cuando el precio ya está muy arriba en su banda de 6 meses?
Prueba distintos topes sobre la Automática fijada (la regla se elige al comprar y se mantiene hasta vender).
Escribe data/investigacion_tope.txt."""
import time, json
import numpy as np
from investigacion import cargar, backtest, sma, E, dia, PERIODOS
from investigacion_niveles import pos_rango, en_rango, maq, resumen

COMP, VEND, STOP, NV = 0.2, 0.6, 0.08, 120
TOPES = (1.01, 0.95, 0.9, 0.8, 0.7)   # 1.01 = sin tope (la versión actual)

def fijada(fl, r, cr, pos, tope):
    n = len(cr); sen = np.zeros(n, bool); st = False; modo = ""
    for i in range(n):
        if not st:
            if fl[i] and r[i]: st, modo = True, "R"
            elif (not fl[i]) and cr[i] and not (pos[i] > tope): st, modo = True, "T"
        else:
            if modo == "T" and not cr[i]: st = False
            elif modo == "R" and not r[i]: st = False
        sen[i] = st
    return sen

def nombre(t): return "Automática fijada, sin tope (actual)" if t > 1 else f"Automática fijada, no compra por tendencia si el precio está sobre el {int(round(t*100))}% de su banda de 6 meses"

def correr(data, periodos=PERIODOS):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl=en_rango(C), pos=pos_rango(C, NV))
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            i0 = int(np.searchsorted(d, ini)); fl = v["fl"][:k]; cr = v["cr"][:k]; pos = v["pos"][:k]
            def bt(sen): return backtest(d, O, C, sen, lambda i, h, pp, sen=sen: not sen[i], ini)
            c = bt(cr); res = {"Comprar y mantener": dict(ret=c["hold"], hold=c["hold"], expo=1.0, trades=[]), "Cruce 50/200": c}
            r = maq(d, O, C, pos, ini, COMP, VEND, STOP) > 0
            for t in TOPES:
                x = bt(fijada(fl, r, cr, pos, t)); x["enrango"] = float(fl[i0:].mean()); res[nombre(t)] = x
            for nom, x in res.items(): acum.setdefault(nom, []).append(x)
        n_ok = len(acum.get("Cruce 50/200", []))
        if n_ok < 10: continue
        for nom, R in acum.items():
            azar = float(np.mean([(1 + x["hold"]) ** x["expo"] - 1 for x in R])); ret = float(np.mean([x["ret"] for x in R]))
            filas.append(dict(periodo=pn, acciones=n_ok, estrategia=nom, ret=ret, vs_azar=ret - azar,
                              gana_azar=float(np.mean([x["ret"] > (1 + x["hold"]) ** x["expo"] - 1 for x in R])),
                              ops=float(np.mean([len(x["trades"]) for x in R])), enrango=float(np.mean([x.get("enrango", 0) for x in R]))))
    return filas

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s", flush=True)
    filas = correr(data)
    json.dump(filas, open("data/investigacion_tope.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_tope.txt", "w", encoding="utf-8").write(txt); print(txt)
