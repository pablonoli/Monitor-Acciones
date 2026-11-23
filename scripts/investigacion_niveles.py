"""Investigación (no es parte del dashboard): ¿qué niveles de compra y venta conviene usar en la estrategia de rango?
Usa el detector de rango actual del dashboard (500 ruedas, ≥3 viajes entre zonas de 20%, cambio < 25%, inclinación < 30%)
y prueba distintos niveles de compra/venta, cortes de pérdida y largos de banda, en varios períodos desde 2018.
Se corre sola cuando cambia este archivo o a mano desde Actions. Escribe data/investigacion_niveles.txt."""
import time, json
import numpy as np
from investigacion import cargar, backtest, sma, E, dia, PERIODOS

# (compra bajo X de la banda, vende sobre Y, corte de pérdida, ruedas de la banda)
COMBOS = [(c, v, 0.08, 120) for c in (0.1, 0.2, 0.3) for v in (0.4, 0.5, 0.6, 0.7, 0.8)]
COMBOS += [(0.2, 0.6, s, 120) for s in (0.05, 0.12, 0.99)] + [(0.2, 0.6, 0.08, nv) for nv in (60, 250)]
VENTANAS = sorted({c[3] for c in COMBOS})

def pos_rango(C, N):
    n = len(C); p = np.full(n, np.nan)
    for i in range(N, n):
        w = C[i - N:i + 1]; p[i] = (C[i] - w.min()) / (w.max() - w.min() + 1e-9)
    return p

def en_rango(C, N=500, K=3, Z=0.2):
    """Igual que enRangoInfo del dashboard."""
    n = len(C); fl = np.zeros(n, bool); x = np.arange(N + 1)
    for i in range(N, n):
        w = C[i - N:i + 1]; mn, mx = w.min(), w.max()
        if abs(C[i] / C[i - N] - 1) >= 0.25: continue
        q = (w - mn) / (mx - mn + 1e-9); z = np.where(q <= Z, -1, np.where(q >= 1 - Z, 1, 0)); z = z[z != 0]
        if len(z) < 2 or (z[1:] != z[:-1]).sum() < K: continue
        co = np.polyfit(x, np.log(w), 1); res = np.log(w) - np.polyval(co, x)
        fl[i] = abs(co[0] * N) / (res.std() * 4 + 1e-9) < 0.3
    return fl

def maq(d, O, C, p, ini, comp, vend, stop):
    n = len(C); sen = np.zeros(n, int); st = 0; ent = 0.0; bl = -1
    for i in range(n):
        if i + 1 < n and d[i + 1] < ini: continue
        if st == 0:
            if p[i] < comp and i > bl: st = 1; ent = O[i + 1] if i + 1 < n else C[i]
        else:
            s = C[i] <= ent * (1 - stop)
            if p[i] > vend or s:
                st = 0
                if s: bl = i + 10
        sen[i] = st
    return sen

def correr(data, periodos=PERIODOS):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl=en_rango(C), pos={N: pos_rango(C, N) for N in VENTANAS})
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            i0 = int(np.searchsorted(d, ini)); fl = v["fl"][:k]
            def bt(sen): return backtest(d, O, C, sen, lambda i, h, pp, sen=sen: not sen[i], ini)
            cr = bt(v["cr"][:k]); res = {"Comprar y mantener": dict(ret=cr["hold"], hold=cr["hold"], expo=1.0, trades=[]), "Cruce 50/200": cr}
            for (c, ve, st, nv) in COMBOS:
                r = maq(d, O, C, v["pos"][nv][:k], ini, c, ve, st) > 0
                x = bt(np.where(fl, r, v["cr"][:k])); x["enrango"] = float(fl[i0:].mean())
                res[f"Auto: compra <{int(c*100)}%, venta >{int(ve*100)}%, corte {'sin' if st > .5 else str(int(st*100)) + '%'}, banda {nv} ruedas"] = x
            for nom, x in res.items(): acum.setdefault(nom, []).append(x)
        n_ok = len(acum.get("Cruce 50/200", []))
        if n_ok < 10: continue
        for nom, R in acum.items():
            azar = float(np.mean([(1 + x["hold"]) ** x["expo"] - 1 for x in R])); ret = float(np.mean([x["ret"] for x in R]))
            filas.append(dict(periodo=pn, acciones=n_ok, estrategia=nom, ret=ret, vs_azar=ret - azar,
                              gana_azar=float(np.mean([x["ret"] > (1 + x["hold"]) ** x["expo"] - 1 for x in R])),
                              ops=float(np.mean([len(x["trades"]) for x in R])), enrango=float(np.mean([x.get("enrango", 0) for x in R]))))
    return filas

def resumen(filas):
    out = []; clave = None
    for f in filas:
        if f["periodo"] != clave:
            clave = f["periodo"]; out.append(f"\n== {f['periodo']} ({f['acciones']} acciones)")
        out.append(f"  {f['estrategia']:72s} rinde {100*f['ret']:7.1f}%  vs azar {100*f['vs_azar']:+6.1f}  gana al azar en {100*f['gana_azar']:3.0f}%  ops {f['ops']:4.1f}  en rango {100*f['enrango']:3.0f}% del tiempo")
    # Resumen: promedio de "vs azar" en los 5 períodos separados y en cuántos le gana al cruce 50/200
    sep = ("2018-2019", "2020 (covid)", "2021", "2022 (bajista)", "2023 a hoy")
    por = {}
    for f in filas:
        if f["periodo"] in sep: por.setdefault(f["estrategia"], {})[f["periodo"]] = f
    out.append("\n== RESUMEN: promedio de los 5 períodos separados (vs azar) y en cuántos rinde más que el cruce 50/200")
    cruce = por.get("Cruce 50/200", {})
    tabla = []
    for nom, d in por.items():
        if nom in ("Comprar y mantener",): continue
        pr = np.mean([x["vs_azar"] for x in d.values()]); gan = sum(1 for p, x in d.items() if x["ret"] > cruce.get(p, {"ret": 9e9})["ret"])
        tabla.append((pr, nom, gan, len(d)))
    for pr, nom, gan, n in sorted(tabla, reverse=True): out.append(f"  {nom:72s} vs azar promedio {100*pr:+6.1f}   mejor que el cruce en {gan} de {n} períodos")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s", flush=True)
    filas = correr(data)
    json.dump(filas, open("data/investigacion_niveles.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_niveles.txt", "w", encoding="utf-8").write(txt); print(txt)
