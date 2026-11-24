"""Investigación (no es parte del dashboard): ¿conviene dejar de ver "rango" antes cuando la acción ya se va para arriba?
(caso MA: cruce alcista el 14/08 con el precio en 569, pero la marca de rango siguió hasta el 22/10 y compró el 19/11 a 637).
Escribe data/investigacion_salir_rango.txt."""
import time, json
import numpy as np
from investigacion import cargar, sma, E, dia, PERIODOS
from investigacion_niveles import pos_rango, en_rango, maq
from investigacion_reciente import fijada, reciente
from investigacion_parciales import bt_frac, texto

COMP, VEND, STOP, NV = 0.2, 0.6, 0.08, 120
VARIANTES = ["Actual (rango con memoria + espera de 4 semanas al salir)",
             "Sin la espera de 4 semanas",
             "Sale del rango apenas hay cruce alcista y el precio está sobre la media 200 (con espera)",
             "Sale del rango apenas hay cruce alcista y el precio está sobre la media 200 (sin espera)",
             "Sale del rango con cruce alcista y el precio 10% sobre la media 200 (sin espera)"]

def correr(data, periodos=PERIODOS):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        s200 = sma(C, 200); fl = en_rango(C); cr = E(sma(C, 50) > s200)
        fe2 = fl & ~(cr & E(C > s200)); fe4 = fl & ~(cr & E(C > 1.10 * s200))
        pre[s] = dict(d=d, O=O, C=C, cr=cr, pos=pos_rango(C, NV), fl=[fl, fl, fe2, fe2, fe4], rec=[reciente(f, 20) for f in (fl, fl, fe2, fe2, fe4)])
    brake = [True, False, True, False, False]
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            cr, pos = v["cr"][:k], v["pos"][:k]
            r = maq(d, O, C, pos, ini, COMP, VEND, STOP) > 0
            res = {"Comprar y mantener": bt_frac(d, O, C, np.ones(k), ini), "Cruce 50/200": bt_frac(d, O, C, cr.astype(float), ini)}
            for j, nom in enumerate(VARIANTES):
                fl = v["fl"][j][:k]; rec = v["rec"][j][:k] if brake[j] else np.zeros(k, bool)
                res[nom] = bt_frac(d, O, C, fijada(fl, r, cr, pos, rec, 0.6 if brake[j] else None).astype(float), ini)
            for nom, x in res.items(): acum.setdefault(nom, []).append(x)
        n_ok = len(acum.get("Cruce 50/200", []))
        if n_ok < 10: continue
        hold = np.array([x["hold"] for x in acum["Comprar y mantener"]]); mask = hold <= np.quantile(hold, 0.9)
        base = np.array([x["ret"] for x in acum[VARIANTES[0]]])
        for nom, R in acum.items():
            rets = np.array([x["ret"] for x in R]); dds = np.array([x["dd"] for x in R])
            azar = float(np.mean([(1 + x["hold"]) ** x["expo"] - 1 for x in R]))
            filas.append(dict(periodo=pn, acciones=n_ok, estrategia=nom, ret=float(rets.mean()), vs_azar=float(rets.mean()) - azar, dd=float(dds.mean()), expo=float(np.mean([x["expo"] for x in R])),
                              ops=float(np.mean([x["ops"] for x in R])), mediana=float(np.median(rets)), sin_top=float(rets[mask].mean()), dd_med=float(np.median(dds)), mejor=float(np.mean(rets > base + 1e-9)), peor=float(np.mean(rets < base - 1e-9))))
    return filas

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s", flush=True)
    filas = correr(data)
    json.dump(filas, open("data/investigacion_salir_rango.json", "w"), ensure_ascii=False)
    txt = texto(filas); open("data/investigacion_salir_rango.txt", "w", encoding="utf-8").write(txt); print(txt)
