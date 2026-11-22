"""Investigación (no es parte del dashboard): ¿qué horizonte conviene para detectar que una acción está en rango?
Prueba la estrategia Automática (rango cuando la acción está en rango, cruce 50/200 si no) con distintas ventanas de
detección y distintos largos de piso/techo, en varios períodos desde 2018. Se corre sola cuando cambia este archivo
o a mano desde Actions. Escribe data/investigacion_rango.txt."""
import sys, time, json
import numpy as np
from investigacion import cargar, backtest, sma, E, dia, PERIODOS

# (ventana de detección en ruedas, mínimo de cruces del promedio)
# (ventana, mínimo de cruces, condición extra): "ret" = en 120 ruedas se movió menos de 15%; "media" = el precio está a menos de 10% de su promedio de la ventana
DETECTORES = [(180, 6, "ambas"), (250, 6, "neto"), (360, 8, "neto"), (500, 8, "neto"), (500, 8, None)]   # "neto": en toda la ventana el precio terminó a menos de 25% de donde empezó
PISOS = [120]          # largo de la ventana para ubicar piso y techo

def pos_rango(C, N):
    n = len(C); p = np.full(n, np.nan)
    for i in range(N, n):
        w = C[i - N:i + 1]; p[i] = (C[i] - w.min()) / (w.max() - w.min() + 1e-9)
    return p

def en_rango(C, N, K, cond=None):
    n = len(C); fl = np.zeros(n, bool); x = np.arange(N + 1)
    for i in range(N, n):
        w = C[i - N:i + 1]; s = np.sign(w - w.mean())
        if (s[1:] != s[:-1]).sum() < K: continue
        co = np.polyfit(x, np.log(w), 1); res = np.log(w) - np.polyval(co, x)
        ok = abs(co[0] * N) / (res.std() * 4 + 1e-9) < 0.3
        if ok and cond in ("ret", "ambas") and i >= 120: ok = abs(C[i] / C[i - 120] - 1) < 0.15
        if ok and cond in ("media", "ambas"): ok = abs(C[i] / w.mean() - 1) < 0.10
        if ok and cond == "neto": ok = abs(C[i] / C[i - N] - 1) < 0.25
        fl[i] = ok
    return fl

def maq(d, O, C, p, ini):
    n = len(C); sen = np.zeros(n, int); st = 0; ent = 0.0; bl = -1
    for i in range(n):
        if i + 1 < n and d[i + 1] < ini: continue
        if st == 0:
            if p[i] < 0.2 and i > bl: st = 1; ent = O[i + 1] if i + 1 < n else C[i]
        else:
            stop = C[i] <= ent * 0.92
            if p[i] > 0.6 or stop:
                st = 0
                if stop: bl = i + 10
        sen[i] = st
    return sen

def correr(data, periodos=PERIODOS):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl={nk: en_rango(C, *nk) for nk in DETECTORES}, pos={N: pos_rango(C, N) for N in PISOS})
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9
        acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            i0 = int(np.searchsorted(d, ini))
            def bt(sen): return backtest(d, O, C, sen, lambda i, h, pp, sen=sen: not sen[i], ini)
            res = {"Comprar y mantener": None, "Cruce 50/200": bt(v["cr"][:k])}
            for nk in DETECTORES:
                fl = v["fl"][nk][:k]
                for N in PISOS:
                    r = maq(d, O, C, v["pos"][N][:k], ini) > 0
                    x = bt(np.where(fl, r, v["cr"][:k])); x["enrango"] = float(fl[i0:].mean())
                    res[f"Auto: {nk[0]} ruedas, ≥{nk[1]} cruces, extra={nk[2] or 'ninguna'}"] = x
            res["Comprar y mantener"] = dict(ret=res["Cruce 50/200"]["hold"], hold=res["Cruce 50/200"]["hold"], expo=1.0, trades=[], dd=0.0)
            for nom, x in res.items(): acum.setdefault(nom, []).append(x)
        n_ok = len(acum.get("Cruce 50/200", []))
        if n_ok < 10: continue
        for nom, R in acum.items():
            azar = float(np.mean([(1 + x["hold"]) ** x["expo"] - 1 for x in R]))
            ret = float(np.mean([x["ret"] for x in R]))
            filas.append(dict(periodo=pn, acciones=n_ok, estrategia=nom, ret=ret, vs_azar=ret - azar,
                              gana_azar=float(np.mean([x["ret"] > (1 + x["hold"]) ** x["expo"] - 1 for x in R])),
                              ops=float(np.mean([len(x["trades"]) for x in R])), enrango=float(np.mean([x.get("enrango", 0) for x in R]))))
    return filas

def resumen(filas):
    out = []; clave = None
    for f in filas:
        if f["periodo"] != clave:
            clave = f["periodo"]; out.append(f"\n== {f['periodo']} ({f['acciones']} acciones)")
        out.append(f"  {f['estrategia']:62s} rinde {100*f['ret']:7.1f}%  vs azar {100*f['vs_azar']:+6.1f}  gana al azar en {100*f['gana_azar']:3.0f}%  ops {f['ops']:4.1f}  en rango {100*f['enrango']:3.0f}% del tiempo")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s", flush=True)
    filas = correr(data)
    json.dump(filas, open("data/investigacion_rango.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_rango.txt", "w", encoding="utf-8").write(txt); print(txt)
