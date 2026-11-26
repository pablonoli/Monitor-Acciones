"""Consulta: últimas señales Strong (círculo verde + Z > 2,5 + volumen > 1,5x) en la lista del dashboard.
Misma regla que tendencia.py. Solo escribe data/consulta_ultimas_strong.txt en esta rama."""
import json
import sys

import pandas as pd

sys.path.insert(0, "scripts")
import tendencia as T  # noqa: E402

cfg = json.load(open(T.RUTA_TICKERS))
favoritas = json.load(open("data/favoritas.json"))
acciones = [a for a in dict.fromkeys(favoritas + cfg.get("extras", [])) if a != cfg["filtro"]]

datos = {}
for i in range(0, len(acciones), T.LOTE):
    datos.update(T.bajar_lote(acciones[i:i + T.LOTE]))
for s in acciones:
    if s not in datos:
        x = T.bajar_uno(s)
        if x is not None:
            datos[s] = x

filas = []
for s, df in datos.items():
    x = df[["High", "Low", "Close", "Volume"]].dropna(subset=["High", "Low", "Close"])
    if len(x) < 205:
        continue
    c, h, lo = x["Close"], x["High"], x["Low"]
    techo = h.shift(1).rolling(20).max()
    mom = c.diff()
    z = (mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)
    verde = (c > techo) & (c.shift(1) <= techo.shift(1)) & (z > 1.5) & (c > c.rolling(200).mean())
    strong = verde & (z > 2.5) & (x["Volume"] > 1.5 * x["Volume"].rolling(20).mean())
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atrp = (tr / c).rolling(14).mean()
    stop = lo.rolling(20).min()
    for i in [k for k in range(len(x)) if bool(strong.iloc[k])]:
        abre = float(df["Open"].iloc[i + 1]) if i + 1 < len(x) else None
        filas.append(dict(fecha=x.index[i].date(), s=s, cierre=float(c.iloc[i]), z=float(z.iloc[i]), atrp=float(atrp.iloc[i]),
                          stop=float(stop.iloc[i]), abre=abre, hoy=float(c.iloc[-1]), ruedas=len(x) - 1 - i))

filas.sort(key=lambda r: r["fecha"], reverse=True)
import datetime as dt
desde, hasta = dt.date(2026, 1, 1), dt.date(2026, 8, 31)
per = sorted([r for r in filas if desde <= r["fecha"] <= hasta], key=lambda r: (r["fecha"], r["s"]))
def plan_a(s, i_sig, stop):
    df = datos[s].dropna(subset=["Open", "High", "Low", "Close"])
    if i_sig + 1 >= len(df):
        return None, ""
    e = float(df["Open"].iloc[i_sig + 1]); R = e - stop
    if R <= 0:
        return None, "abrió debajo del stop"
    mitad, st, res = False, stop, 0.0
    for k in range(i_sig + 1, min(i_sig + 21, len(df))):
        o, h, lo, c = (float(df[col].iloc[k]) for col in ("Open", "High", "Low", "Close"))
        peso = 0.5 if mitad else 1.0
        if lo <= st:
            sal = min(o, st)
            return (res + peso * (sal - e) / R, R / e), "stop" if not mitad else "mitad en +2R y resto al costo"
        if not mitad and h >= e + 2 * R:
            res += 0.5 * 2; mitad = True; st = e
        if k == i_sig + 20:
            return (res + (0.5 if mitad else 1.0) * (c - e) / R, R / e), "rueda 20" + (" (mitad en +2R)" if mitad else "")
    return (res + (0.5 if mitad else 1.0) * (float(df["Close"].iloc[-1]) - e) / R, R / e), "abierta"
lin = ["Señales Strong de enero a agosto 2026 (lista del dashboard). Resultado con Plan A, sin costos, en R (1R = el riesgo, $40.000 en tu ejemplo).",
       "Ojo: es cada señal por separado; con $4.000.000 no alcanza para tomarlas todas si caen juntas.", "",
       "fecha       acción   Z     ATR%  opc7  stop%   ganancia%  resultado  en pesos     cómo terminó"]
tot = tot7 = 0; n7 = 0
for r in per:
    df = datos[r["s"]].dropna(subset=["Open", "High", "Low", "Close"])
    i = list(df.index.date).index(r["fecha"])
    rr, como = plan_a(r["s"], i, r["stop"])
    res, dist = rr if rr else (None, None)
    pct = res * dist * 100 if res is not None else None
    pesos = res * 40000 if res is not None else None
    o7 = r["atrp"] >= 0.02
    if res is not None:
        tot += res
        if o7: tot7 += res; n7 += 1
    lin.append(f"{r['fecha']}  {r['s']:<6} {r['z']:>4.2f}  {r['atrp']*100:>4.1f}%  {'sí' if o7 else 'no':<4}  {(dist*100 if dist else 0):>5.1f}%  {('%+.1f%%' % pct) if pct is not None else '-':>8}  {('%+.2fR' % res) if res is not None else '-':>8}  {('%+10.0f' % pesos) if pesos is not None else '-':>10}   {como}")
lin += ["", "ganancia% = cuánto ganó o perdió la plata puesta en esa operación (con la venta de la mitad en +2R si pasó). stop% = distancia de la compra al stop.", "en pesos = con $4.000.000 y 1% de riesgo ($40.000 por operación).", f"Total: {len(per)} señales, suma {tot:+.1f}R. Solo opción 7 (ATR >= 2%): {n7} señales, suma {tot7:+.1f}R."]
open("data/consulta_strong_ene_ago.txt", "w").write("\n".join(lin) + "\n")
print("\n".join(lin))
out = ["Últimas 25 señales Strong (lista del dashboard). Último dato: " + str(max(d.index[-1].date() for d in datos.values())), ""]
out.append("fecha       acción  cierre    Z     ATR%  opc7  compra(apertura sig.)  stop(mín20)  precio hoy  var% desde compra  ruedas")
for r in filas[:25]:
    var = (r["hoy"] / r["abre"] - 1) * 100 if r["abre"] else float("nan")
    out.append(f"{r['fecha']}  {r['s']:<6} {r['cierre']:>8.2f}  {r['z']:>4.2f}  {r['atrp']*100:>4.1f}%  {'sí' if r['atrp'] >= 0.02 else 'no':<4}  "
               f"{(r['abre'] or float('nan')):>10.2f}            {r['stop']:>9.2f}   {r['hoy']:>9.2f}   {var:>+7.1f}%           {r['ruedas']}")
open("data/consulta_ultimas_strong.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))


# ---- Detalle de cada operación: fechas y precios de compra y venta ----
def detalle(s, i_sig, stop):
    df = datos[s].dropna(subset=["Open", "High", "Low", "Close"])
    if i_sig + 1 >= len(df):
        return None
    f = lambda k: df.index[k].date().strftime("%d/%m")
    e = float(df["Open"].iloc[i_sig + 1]); R = e - stop
    if R <= 0:
        return None
    d = dict(compra=f(i_sig + 1), pc=e, stop=stop, obj=e + 2 * R, mitad=None)
    mitad, st, res = False, stop, 0.0
    for k in range(i_sig + 1, min(i_sig + 21, len(df))):
        o, h, lo, c = (float(df[col].iloc[k]) for col in ("Open", "High", "Low", "Close"))
        if lo <= st:
            sal = min(o, st)
            d.update(venta=f(k), pv=sal, como=("tocó el stop" if not mitad else "volvió al precio de compra (stop movido)"))
            res += (0.5 if mitad else 1.0) * (sal - e) / R
            d["R"] = res; return d
        if not mitad and h >= e + 2 * R:
            res += 1.0; mitad = True; st = e; d["mitad"] = (f(k), e + 2 * R)
        if k == i_sig + 20:
            res += (0.5 if mitad else 1.0) * (c - e) / R
            d.update(venta=f(k), pv=c, como="cierre en la rueda 20"); d["R"] = res; return d
    d.update(venta="abierta", pv=float(df["Close"].iloc[-1]), como="sigue abierta")
    d["R"] = res + (0.5 if mitad else 1.0) * (d["pv"] - e) / R
    return d

det = []
for r in per:
    df = datos[r["s"]].dropna(subset=["Open", "High", "Low", "Close"])
    i = list(df.index.date).index(r["fecha"])
    d = detalle(r["s"], i, r["stop"])
    if d:
        d["s"] = r["s"]; d["senal"] = r["fecha"].strftime("%d/%m"); det.append(d)

def txt(d):
    tam = 40000 / ((d["pc"] - d["stop"]) / d["pc"])
    pct = d["R"] * (d["pc"] - d["stop"]) / d["pc"] * 100
    l = [f"{d['s']}  señal {d['senal']}  ->  {d['R']*40000:+,.0f} pesos ({pct:+.1f}%)".replace(",", "."),
         f"   Compra {d['compra']} a US$ {d['pc']:.2f} (posición de unos ${tam:,.0f}). Stop US$ {d['stop']:.2f}, objetivo +2R US$ {d['obj']:.2f}.".replace(",", ".")]
    if d["mitad"]:
        l.append(f"   Vende la mitad el {d['mitad'][0]} a US$ {d['mitad'][1]:.2f} (+2R) y sube el stop al precio de compra.")
    l.append(f"   Vende {'el resto' if d['mitad'] else 'todo'} el {d['venta']} a US$ {d['pv']:.2f}: {d['como']}.")
    return "\n".join(l)

orden = sorted(det, key=lambda d: -d["R"])
sal = ["LAS 10 MÁS GANADORAS Y LAS 10 MÁS PERDEDORAS (Strong, enero a agosto 2026, Plan A, $40.000 de riesgo, sin costos)", "",
       "=== 10 MÁS GANADORAS ==="] + [txt(d) + "\n" for d in orden[:10]] + ["=== 10 MÁS PERDEDORAS ==="] + [txt(d) + "\n" for d in orden[::-1][:10]]
sal += ["=== TODAS, POR FECHA ==="] + [txt(d) + "\n" for d in det]
open("data/consulta_strong_detalle.txt", "w").write("\n".join(sal) + "\n")
print("\n".join(sal[:60]))


# ---- Plan D: en la rueda 20, si va +1R o más, sigue hasta la rueda 40 ----
def detalle_d(s, i_sig, stop):
    df = datos[s].dropna(subset=["Open", "High", "Low", "Close"])
    if i_sig + 1 >= len(df):
        return None
    f = lambda k: df.index[k].date().strftime("%d/%m")
    e = float(df["Open"].iloc[i_sig + 1]); R = e - stop
    if R <= 0:
        return None
    d = dict(compra=f(i_sig + 1), pc=e, mitad=None, sigue=False)
    mitad, st, res, tope = False, stop, 0.0, i_sig + 20
    k = i_sig + 1
    while k < len(df):
        o, h, lo, c = (float(df[col].iloc[k]) for col in ("Open", "High", "Low", "Close"))
        if lo <= st:
            sal = min(o, st)
            res += (0.5 if mitad else 1.0) * (sal - e) / R
            d.update(venta=f(k), pv=sal, ruedas=k - i_sig, como="tocó el stop" if not mitad else "volvió al precio de compra"); d["R"] = res; return d
        if not mitad and h >= e + 2 * R:
            res += 1.0; mitad = True; st = e; d["mitad"] = (f(k), e + 2 * R)
        if k == tope:
            if tope == i_sig + 20 and (c - e) / R >= 1:
                tope = i_sig + 40; d["sigue"] = True
            else:
                res += (0.5 if mitad else 1.0) * (c - e) / R
                d.update(venta=f(k), pv=c, ruedas=k - i_sig, como=f"cierre en la rueda {k - i_sig}"); d["R"] = res; return d
        k += 1
    d.update(venta="abierta", pv=float(df["Close"].iloc[-1]), ruedas=len(df) - 1 - i_sig, como="sigue abierta")
    d["R"] = res + (0.5 if mitad else 1.0) * (d["pv"] - e) / R
    return d

comp = []
for r, a in zip(per, [None] * len(per)):
    df = datos[r["s"]].dropna(subset=["Open", "High", "Low", "Close"])
    i = list(df.index.date).index(r["fecha"])
    A = detalle(r["s"], i, r["stop"]); D = detalle_d(r["s"], i, r["stop"])
    if A and D:
        dist = (A["pc"] - r["stop"]) / A["pc"]
        comp.append(dict(s=r["s"], senal=r["fecha"].strftime("%d/%m"), A=A, D=D, dist=dist))
p = lambda x: f"{x*40000:+,.0f}".replace(",", ".")
sig = [c for c in comp if c["D"]["sigue"]]
out = ["PLAN A vs PLAN D en las Strong de enero a agosto 2026 ($40.000 de riesgo, sin costos)",
       "Plan D: igual que A, pero si en la rueda 20 va +1R o más, sigue hasta la rueda 40 (o hasta el stop).", "",
       f"Operaciones que siguieron hasta la 40: {len(sig)} de {len(comp)}", "",
       "acción señal  compra          venta A               venta D                        A (%/pesos)        D (%/pesos)       diferencia"]
for c in sig:
    A, D = c["A"], c["D"]
    out.append(f"{c['s']:<5} {c['senal']}  {A['compra']} a {A['pc']:.2f}  {A['venta']} a {A['pv']:.2f}   {D['venta']} a {D['pv']:.2f} ({D['como']})"
               f"   {A['R']*c['dist']*100:+.1f}% {p(A['R'])}   {D['R']*c['dist']*100:+.1f}% {p(D['R'])}   {p(D['R']-A['R'])}")
tA = sum(c["A"]["R"] for c in comp); tD = sum(c["D"]["R"] for c in comp)
mej = sum(1 for c in sig if c["D"]["R"] > c["A"]["R"] + 1e-9)
out += ["", f"De las {len(sig)} que siguieron: {mej} mejoraron y {len(sig) - mej} empeoraron.",
        f"Total Plan A: {tA:+.1f}R = {p(tA)} pesos.   Total Plan D: {tD:+.1f}R = {p(tD)} pesos.   Diferencia: {p(tD - tA)}."]
open("data/consulta_strong_plan_d.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
