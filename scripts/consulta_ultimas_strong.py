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
            return res + peso * (sal - e) / R, "stop" if not mitad else "mitad en +2R y resto al costo"
        if not mitad and h >= e + 2 * R:
            res += 0.5 * 2; mitad = True; st = e
        if k == i_sig + 20:
            return res + (0.5 if mitad else 1.0) * (c - e) / R, "rueda 20" + (" (mitad en +2R)" if mitad else "")
    return res + (0.5 if mitad else 1.0) * (float(df["Close"].iloc[-1]) - e) / R, "abierta"
lin = ["Señales Strong de enero a agosto 2026 (lista del dashboard). Resultado con Plan A, sin costos, en R (1R = el riesgo, $40.000 en tu ejemplo).",
       "Ojo: es cada señal por separado; con $4.000.000 no alcanza para tomarlas todas si caen juntas.", "",
       "fecha       acción   Z     ATR%  opc7  resultado   cómo terminó"]
tot = tot7 = 0; n7 = 0
for r in per:
    df = datos[r["s"]].dropna(subset=["Open", "High", "Low", "Close"])
    i = list(df.index.date).index(r["fecha"])
    res, como = plan_a(r["s"], i, r["stop"])
    o7 = r["atrp"] >= 0.02
    if res is not None:
        tot += res
        if o7: tot7 += res; n7 += 1
    lin.append(f"{r['fecha']}  {r['s']:<6} {r['z']:>4.2f}  {r['atrp']*100:>4.1f}%  {'sí' if o7 else 'no':<4}  {('%+.2fR' % res) if res is not None else '-':>8}   {como}")
lin += ["", f"Total: {len(per)} señales, suma {tot:+.1f}R. Solo opción 7 (ATR >= 2%): {n7} señales, suma {tot7:+.1f}R."]
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
