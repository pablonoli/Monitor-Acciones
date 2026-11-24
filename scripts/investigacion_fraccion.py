"""Investigación (no es parte del dashboard): ¿qué parte vender al llegar a +20% de ganancia (el resto sale con la regla normal)?"""
import time, json
import numpy as np
from investigacion import cargar, sma, E, dia, PERIODOS
from investigacion_niveles import pos_rango, en_rango, maq
from investigacion_parciales import bt_frac, reciente, COMP, VEND, STOP, NV

def simular(fl, r, cr, C, bloq, f_venta, umbral):
    n = len(C); tgt = np.zeros(n); st = False; modo = ""; frac = 0.0; ent = 0.0; esp = False
    for i in range(n):
        if not st:
            if fl[i] and r[i]: st, modo = True, "R"
            elif (not fl[i]) and cr[i] and not bloq[i]: st, modo = True, "T"
            if st: frac = 1.0; ent = C[i]
        else:
            if (modo == "T" and not cr[i]) or (modo == "R" and not r[i]): st = False; frac = 0.0
            elif f_venta > 0 and frac == 1.0 and C[i] >= (1 + umbral) * ent: frac = 1.0 - f_venta
        tgt[i] = frac if st else 0.0
    return tgt

CASOS = [("Actual: todo al cruce", 0, 0.2), ("Vende 25% a +20%", 0.25, 0.2), ("Vende 1/3 a +20%", 1 / 3, 0.2), ("Vende 50% a +20%", 0.5, 0.2), ("Vende 75% a +20%", 0.75, 0.2),
         ("Vende 50% a +30%", 0.5, 0.3), ("Vende 25% a +30%", 0.25, 0.3), ("Vende 25% a +10%", 0.25, 0.1)]

def correr(data):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        fl = en_rango(C); pos = pos_rango(C, NV)
        pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl=fl, pos=pos, bloq=reciente(fl, 20) & E(pos > 0.6))
    filas = []
    for (pn, a, b) in PERIODOS:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            fl, cr, pos, bloq = v["fl"][:k], v["cr"][:k], v["pos"][:k], v["bloq"][:k]
            r = maq(d, O, C, pos, ini, COMP, VEND, STOP) > 0
            acum.setdefault("Comprar y mantener", []).append(bt_frac(d, O, C, np.ones(k), ini))
            for nom, fv, um in CASOS: acum.setdefault(nom, []).append(bt_frac(d, O, C, simular(fl, r, cr, C, bloq, fv, um), ini))
        n_ok = len(acum.get("Comprar y mantener", []))
        if n_ok < 10: continue
        base = np.array([x["ret"] for x in acum[CASOS[0][0]]])
        for nom, R in acum.items():
            rets = np.array([x["ret"] for x in R]); dds = np.array([x["dd"] for x in R])
            filas.append(dict(periodo=pn, acciones=n_ok, estrategia=nom, mediana=float(np.median(rets)), prom=float(rets.mean()), dd=float(np.median(dds)),
                              expo=float(np.mean([x["expo"] for x in R])), mejor=float(np.mean(rets > base + 1e-9)), peor=float(np.mean(rets < base - 1e-9))))
    return filas

def texto(filas):
    out = []; per = []
    for f in filas:
        if f["periodo"] not in per: per.append(f["periodo"])
    for p in per:
        F = [f for f in filas if f["periodo"] == p]; out.append(f"== {p} ({F[0]['acciones']} acciones)")
        for f in F: out.append(f"  {f['estrategia']:26} mediana {100*f['mediana']:6.1f}%  promedio {100*f['prom']:6.1f}%  caída típica {100*f['dd']:6.1f}%  invertido {100*f['expo']:3.0f}%  mejor que actual en {100*f['mejor']:3.0f}%, peor en {100*f['peor']:3.0f}%")
    sep = [p for p in per if p != "2018 a hoy"]; out.append("== RESUMEN: promedio de los períodos separados")
    for n in dict.fromkeys(f["estrategia"] for f in filas):
        R = [f for f in filas if f["estrategia"] == n and f["periodo"] in sep]
        out.append(f"  {n:26} mediana {100*np.mean([f['mediana'] for f in R]):6.1f}%  promedio {100*np.mean([f['prom'] for f in R]):6.1f}%  caída típica {100*np.mean([f['dd'] for f in R]):6.1f}%")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos en", round(time.time() - t0), "s", flush=True)
    filas = correr(data); txt = texto(filas)
    open("data/investigacion_fraccion.txt", "w", encoding="utf-8").write(txt); json.dump(filas, open("data/investigacion_fraccion.json", "w"), ensure_ascii=False); print(txt)
