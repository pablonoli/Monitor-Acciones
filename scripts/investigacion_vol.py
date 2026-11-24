"""Investigación (no es parte del dashboard): vender una parte a +20% según qué tan rápido se mueve la acción (volatilidad de los 12 meses ANTES de empezar)."""
import time
import numpy as np
from investigacion import cargar, sma, E, dia, PERIODOS
from investigacion_niveles import pos_rango, en_rango, maq
from investigacion_parciales import bt_frac, reciente, COMP, VEND, STOP, NV
from investigacion_fraccion import simular

CASOS = [("Actual: todo al cruce", 0), ("Vende 1/3 a +20%", 1 / 3), ("Vende 50% a +20%", 0.5)]
GRUPOS = [("LENTAS (vol. menor a 30%)", 0, 0.30), ("MEDIAS (30% a 45%)", 0.30, 0.45), ("RÁPIDAS (más de 45%)", 0.45, 9)]

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos en", round(time.time() - t0), "s", flush=True)
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        fl = en_rango(C); pos = pos_rango(C, NV)
        pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl=fl, pos=pos, bloq=reciente(fl, 20) & E(pos > 0.6))
    out = []
    for (pn, a, b) in PERIODOS:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            i0 = int(np.searchsorted(d, ini)); rt = np.diff(np.log(C[max(0, i0 - 253):i0])); vol = float(rt.std() * np.sqrt(252)) if len(rt) > 100 else None
            if vol is None: continue
            gname = next(g[0] for g in GRUPOS if g[1] <= vol < g[2])
            fl, cr, pos, bloq = v["fl"][:k], v["cr"][:k], v["pos"][:k], v["bloq"][:k]
            r = maq(d, O, C, pos, ini, COMP, VEND, STOP) > 0
            for nom, fv in CASOS: acum.setdefault((gname, nom), []).append(bt_frac(d, O, C, simular(fl, r, cr, C, bloq, fv, 0.2), ini))
        out.append(f"== {pn}")
        for g in GRUPOS:
            if (g[0], CASOS[0][0]) not in acum: continue
            base = np.array([x["ret"] for x in acum[(g[0], CASOS[0][0])]])
            out.append(f"  {g[0]} · {len(base)} acciones")
            for nom, _ in CASOS:
                R = acum[(g[0], nom)]; rets = np.array([x["ret"] for x in R]); dds = np.array([x["dd"] for x in R])
                out.append(f"     {nom:24} mediana {100*np.median(rets):6.1f}%  promedio {100*rets.mean():6.1f}%  caída típica {100*np.median(dds):6.1f}%  mejor que actual en {100*np.mean(rets > base + 1e-9):3.0f}%, peor en {100*np.mean(rets < base - 1e-9):3.0f}%")
    txt = "\n".join(out); open("data/investigacion_vol.txt", "w", encoding="utf-8").write(txt); print(txt)
