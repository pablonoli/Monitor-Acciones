"""Investigación (no es parte del dashboard): ventas parciales y salidas más tempranas sobre la Automática actual
(regla fijada al comprar + frenos al salir de un rango). Escribe data/investigacion_parciales.txt."""
import time, json
import numpy as np
from investigacion import cargar, sma, E, dia, PERIODOS, COST
from investigacion_niveles import pos_rango, en_rango, maq

COMP, VEND, STOP, NV = 0.2, 0.6, 0.08, 120
VARIANTES = ["Actual (todo o nada)",
             "T1 vende la mitad al cerrar bajo la media 50",
             "T2 vende la mitad si cae 10% desde el máximo",
             "T3 vende todo al cerrar bajo la media 50 (y vuelve a comprar sobre ella)",
             "T4 vende todo si cae 15% desde el máximo",
             "T5 vende la mitad con +20% de ganancia",
             "T6 escalonada: 1/3 con +20%, 1/3 bajo la media 50, el resto al cruce",
             "R1 (rango) vende la mitad al 40% de la banda y el resto al 60%"]

def reciente(fl, M):
    n = len(fl); c = np.concatenate([[0], np.cumsum(fl.astype(int))]); out = np.zeros(n, bool)
    for i in range(n): out[i] = c[i + 1] - c[max(0, i - M + 1)] > 0
    return out

def simular(fl, r, cr, pos, C, s50, bloq, var):
    n = len(C); tgt = np.zeros(n); st = False; modo = ""; frac = 0.0; ent = 0.0; mx = 0.0; espera = False; pasos = 0
    for i in range(n):
        if not cr[i]: espera = False
        if not st:
            t_ok = (not fl[i]) and cr[i] and not bloq[i]
            if var.startswith("T3"): t_ok = t_ok and C[i] > s50[i]
            if var.startswith("T4"): t_ok = t_ok and not espera
            if fl[i] and r[i]: st, modo = True, "R"
            elif t_ok: st, modo = True, "T"
            if st: frac = 1.0; ent = C[i]; mx = C[i]; pasos = 0
        else:
            mx = max(mx, C[i])
            if modo == "T" and not cr[i]: st = False; frac = 0.0
            elif modo == "R" and not r[i]: st = False; frac = 0.0
            elif modo == "T":
                if var.startswith("T1") and frac == 1 and C[i] < s50[i]: frac = 0.5
                elif var.startswith("T2") and frac == 1 and C[i] <= 0.9 * mx: frac = 0.5
                elif var.startswith("T3") and C[i] < s50[i]: st = False; frac = 0.0
                elif var.startswith("T4") and C[i] <= 0.85 * mx: st = False; frac = 0.0; espera = True
                elif var.startswith("T5") and frac == 1 and C[i] >= 1.2 * ent: frac = 0.5
                elif var.startswith("T6"):
                    if pasos == 0 and C[i] >= 1.2 * ent: frac = 2 / 3; pasos = 1
                    elif pasos <= 1 and C[i] < s50[i] and frac > 1 / 3: frac = 1 / 3; pasos = 2
            elif modo == "R":
                if var.startswith("R1") and frac == 1 and pos[i] > 0.4: frac = 0.5
        tgt[i] = frac if st else 0.0
    return tgt

def bt_frac(days, O, C, tgt, start):
    n = len(C); e = 1.0; p = 0.0; eq = np.ones(n); expo = 0.0; ent = 0; start_i = int(np.searchsorted(days, start))
    for i in range(1, n):
        e *= 1 + p * (O[i] / C[i - 1] - 1); want = p
        if days[i] >= start: want = tgt[i - 1]
        if want != p:
            e *= 1 - COST * abs(want - p)
            if p == 0 and want > 0: ent += 1
            p = want
        e *= 1 + p * (C[i] / O[i] - 1); eq[i] = e
        if days[i] >= start: expo += p
    i0 = max(0, start_i - 1); seg = eq[i0:] / eq[i0]
    return dict(ret=eq[-1] / eq[i0] - 1, expo=expo / max(1, n - start_i), hold=C[-1] / C[i0] - 1, dd=float((seg / np.maximum.accumulate(seg) - 1).min()), ops=ent)

def correr(data, periodos=PERIODOS):
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        fl = en_rango(C); pos = pos_rango(C, NV)
        pre[s] = dict(d=d, O=O, C=C, cr=E(sma(C, 50) > sma(C, 200)), fl=fl, pos=pos, s50=sma(C, 50), bloq=reciente(fl, 20) & E(pos > 0.6))
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
            fl, cr, pos, s50, bloq = v["fl"][:k], v["cr"][:k], v["pos"][:k], v["s50"][:k], v["bloq"][:k]
            r = maq(d, O, C, pos, ini, COMP, VEND, STOP) > 0
            res = {"Comprar y mantener": bt_frac(d, O, C, np.ones(k), ini), "Cruce 50/200": bt_frac(d, O, C, cr.astype(float), ini)}
            for var in VARIANTES: res[var] = bt_frac(d, O, C, simular(fl, r, cr, pos, C, s50, bloq, var), ini)
            for nom, x in res.items(): acum.setdefault(nom, []).append(x)
        n_ok = len(acum.get("Cruce 50/200", []))
        if n_ok < 10: continue
        for nom, R in acum.items():
            azar = float(np.mean([(1 + x["hold"]) ** x["expo"] - 1 for x in R])); ret = float(np.mean([x["ret"] for x in R]))
            filas.append(dict(periodo=pn, acciones=n_ok, estrategia=nom, ret=ret, vs_azar=ret - azar, dd=float(np.mean([x["dd"] for x in R])), expo=float(np.mean([x["expo"] for x in R])), ops=float(np.mean([x["ops"] for x in R]))))
    return filas

def texto(filas):
    out = []; per = []
    for f in filas:
        if f["periodo"] not in per: per.append(f["periodo"])
    for p in per:
        F = [f for f in filas if f["periodo"] == p]; out.append(f"== {p} ({F[0]['acciones']} acciones)")
        for f in F: out.append(f"  {f['estrategia'][:82]:82} rinde {100*f['ret']:7.1f}%  peor caída {100*f['dd']:6.1f}%  vs azar {100*f['vs_azar']:+6.1f}  invertido {100*f['expo']:3.0f}% del tiempo  ops {f['ops']:.1f}")
    sep = [p for p in per if p != "2018 a hoy"]; out.append("== RESUMEN: promedio de los 5 períodos separados")
    base = {p: next(f["ret"] for f in filas if f["periodo"] == p and f["estrategia"] == VARIANTES[0]) for p in sep}
    noms = []
    for f in filas:
        if f["estrategia"] not in noms: noms.append(f["estrategia"])
    for n in noms:
        R = [f for f in filas if f["estrategia"] == n and f["periodo"] in sep]
        mejor = sum(1 for f in R if f["ret"] > base[f["periodo"]] + 1e-9)
        out.append(f"  {n[:82]:82} rinde promedio {100*np.mean([f['ret'] for f in R]):6.1f}%  peor caída promedio {100*np.mean([f['dd'] for f in R]):6.1f}%  rinde más que el actual en {mejor} de 5")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s", flush=True)
    filas = correr(data)
    json.dump(filas, open("data/investigacion_parciales.json", "w"), ensure_ascii=False)
    txt = texto(filas); open("data/investigacion_parciales.txt", "w", encoding="utf-8").write(txt); print(txt)
