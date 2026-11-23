"""Investigación (no es parte del dashboard): ¿conviene fijar la estrategia al comprar?
Compara la Automática actual (cada día usa la regla de rango si está en rango, y el cruce 50/200 si no, así que una operación
puede empezar con una regla y terminar con otra) contra la versión "fijada": se vende con la misma regla con la que se compró.
Escribe data/investigacion_fijar.txt."""
import time, json
import numpy as np
from investigacion import cargar, backtest, sma, E, dia, PERIODOS
from investigacion_niveles import pos_rango, en_rango, maq, resumen

COMP, VEND, STOP, NV = 0.2, 0.6, 0.08, 120

def fijada(fl, r, cr):
    n = len(cr); sen = np.zeros(n, bool); st = False; modo = ""
    for i in range(n):
        if not st:
            if fl[i] and r[i]: st, modo = True, "R"
            elif (not fl[i]) and cr[i]: st, modo = True, "T"
        else:
            if modo == "T" and not cr[i]: st = False
            elif modo == "R" and not r[i]: st = False
        sen[i] = st
    return sen

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
            i0 = int(np.searchsorted(d, ini)); fl = v["fl"][:k]; cr = v["cr"][:k]
            def bt(sen): return backtest(d, O, C, sen, lambda i, h, pp, sen=sen: not sen[i], ini)
            c = bt(cr); res = {"Comprar y mantener": dict(ret=c["hold"], hold=c["hold"], expo=1.0, trades=[]), "Cruce 50/200": c}
            r = maq(d, O, C, v["pos"][:k], ini, COMP, VEND, STOP) > 0
            x = bt(np.where(fl, r, cr)); x["enrango"] = float(fl[i0:].mean()); res["Automática actual (se mezclan las reglas)"] = x
            x = bt(fijada(fl, r, cr)); x["enrango"] = float(fl[i0:].mean()); res["Automática fijada (vende con la regla con que compró)"] = x
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
    json.dump(filas, open("data/investigacion_fijar.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_fijar.txt", "w", encoding="utf-8").write(txt); print(txt)
