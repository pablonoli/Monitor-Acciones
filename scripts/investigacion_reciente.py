"""Investigación (no es parte del dashboard): ¿qué hacer cuando una acción estuvo en rango hace poco y da compra por tendencia?
(caso MSFT: estaba en rango, salió, y en ese momento compró por tendencia en el precio más alto).
Variantes sobre la Automática fijada. Escribe data/investigacion_reciente.txt."""
import time, json
import numpy as np
from investigacion import cargar, backtest, sma, E, dia, PERIODOS
from investigacion_niveles import pos_rango, en_rango, maq, resumen

COMP, VEND, STOP, NV = 0.2, 0.6, 0.08, 120
# (ruedas hacia atrás en que se mira si estuvo en rango, tope de posición en la banda para bloquear: None = bloquea siempre)
VARIANTES = [(0, None), (20, None), (60, None), (120, None), (20, 0.6), (60, 0.6), (120, 0.6), (60, 0.8)]

def reciente(fl, M):
    n = len(fl); c = np.concatenate([[0], np.cumsum(fl.astype(int))]); out = np.zeros(n, bool)
    for i in range(n): out[i] = c[i + 1] - c[max(0, i - M + 1)] > 0
    return out

def fijada(fl, r, cr, pos, rec, tope):
    n = len(cr); sen = np.zeros(n, bool); st = False; modo = ""
    for i in range(n):
        if not st:
            if fl[i] and r[i]: st, modo = True, "R"
            elif (not fl[i]) and cr[i]:
                bloq = rec[i] and (tope is None or pos[i] > tope)
                if not bloq: st, modo = True, "T"
        else:
            if modo == "T" and not cr[i]: st = False
            elif modo == "R" and not r[i]: st = False
        sen[i] = st
    return sen

def nombre(M, t):
    if M == 0: return "Automática fijada, sin cambios (actual)"
    return f"No compra por tendencia si estuvo en rango en las últimas {M} ruedas" + ("" if t is None else f" y el precio está sobre el {int(t*100)}% de su banda")

def correr(data, periodos=PERIODOS):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        fl = en_rango(C); pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl=fl, pos=pos_rango(C, NV), rec={M: reciente(fl, M) for M in {v[0] for v in VARIANTES if v[0]}})
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
            for (M, t) in VARIANTES:
                rec = v["rec"][M][:k] if M else np.zeros(k, bool)
                x = bt(fijada(fl, r, cr, pos, rec, t)); x["enrango"] = float(fl[i0:].mean()); res[nombre(M, t)] = x
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
    json.dump(filas, open("data/investigacion_reciente.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_reciente.txt", "w", encoding="utf-8").write(txt); print(txt)
