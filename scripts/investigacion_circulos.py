"""Investigación: ¿cómo habrían rendido los «círculos verdes» (ver scripts/tendencia.py) de los últimos 6 meses en los CEDEARs?
Universo: las acciones con CEDEAR de data/tendencia_acciones.json (clave "cedear"). Precios en USD de la acción (el CEDEAR la sigue, más el dólar CCL).
Entrada: apertura de la rueda siguiente a la señal. Costo 0,15% por lado. Se compara con comprar cualquier acción cualquier día de la misma ventana (línea de base).
Salidas probadas: a 5/10/20/40 ruedas, hasta hoy, y el plan del simulador (stop por ATR 2,5× o stop amplio; vende la mitad a +2× el riesgo y mueve el stop a la entrada; máx. 60 ruedas)."""
import json, sys
import numpy as np, pandas as pd, yfinance as yf

COSTO = 0.0015
HORIZ = [5, 10, 20, 40]
MAXR = 60


def bajar(simbolos):
    out = {}
    for i in range(0, len(simbolos), 20):
        lote = simbolos[i:i + 20]
        for intento in range(3):
            try:
                d = yf.download(lote, start="2023-01-01", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=False)
                for s in lote:
                    x = d[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
                    if len(x) > 250: out[s] = x
                break
            except Exception as e:
                print("lote falló", intento, e, file=sys.stderr)
    return out


def preparar(x):
    c, h, l, v = x["Close"], x["High"], x["Low"], x["Volume"]
    techo = h.shift(1).rolling(20).max()
    mom = c.diff(); z = (mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)
    verde = (c > techo) & (c.shift(1) <= techo.shift(1)) & (z > 1.5) & (c > c.rolling(200).mean())
    strong = verde & (z > 2.5) & (v > 1.5 * v.rolling(20).mean())
    cant = c.shift(1)
    tr = pd.concat([h - l, (h - cant).abs(), (l - cant).abs()], axis=1).max(axis=1)
    atrp = (tr / c).rolling(14).mean()
    low20 = l.rolling(20).min()
    return dict(O=x["Open"].values, H=h.values, L=l.values, C=c.values, f=x.index, verde=verde.values, strong=strong.values, atrp=atrp.values, low20=low20.values)


def neto(entrada, salida):
    return salida * (1 - COSTO) / (entrada * (1 + COSTO)) - 1


def fijo(P, e, n):
    """Compra en la apertura de e, vende en el cierre de e+n-1 (n ruedas). None si todavía no pasó ese tiempo."""
    j = e + n - 1
    return None if j >= len(P["C"]) else neto(P["O"][e], P["C"][j])


def hasta_hoy(P, e):
    return neto(P["O"][e], P["C"][-1])


def plan(P, e, stop):
    """Devuelve (retorno ponderado, múltiplo de R, motivo, ruedas) o None si el stop no sirve."""
    ent = P["O"][e]
    if not (0 < stop < ent): return None
    R = ent - stop; obj = ent + 2 * R; sc = stop; mitad = False; rm = None; n = len(P["C"])
    for j in range(e, min(n, e + MAXR)):
        o, hi, lo = P["O"][j], P["H"][j], P["L"][j]
        if lo <= sc:
            px = min(sc, o); rr = neto(ent, px)
            if mitad: return 0.5 * rm + 0.5 * rr, 0.5 * 2 + 0.5 * (px - ent) / R, "stop a la entrada", j - e + 1
            return rr, (px - ent) / R, "stop", j - e + 1
        if not mitad and hi >= obj:
            px = max(obj, o); rm = neto(ent, px); mitad = True; sc = ent
    j = min(n, e + MAXR) - 1; rr = neto(ent, P["C"][j]); fin = "abierta" if (j == n - 1 and n - e < MAXR) else "tiempo"
    if mitad: return 0.5 * rm + 0.5 * rr, 0.5 * 2 + 0.5 * (P["C"][j] - ent) / R, fin, j - e + 1
    return rr, (P["C"][j] - ent) / R, fin, j - e + 1


def corto(vals):
    r = resumen(vals or []).split("|")
    return r[1].strip() if len(r) > 1 else "sin datos"


def resumen(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if not len(v): return "sin datos"
    return f"{len(v):>5} ops | promedio {100 * v.mean():+6.2f}% | mediana {100 * np.median(v):+6.2f}% | ganadoras {100 * (v > 0).mean():4.0f}% | mejor {100 * v.max():+6.1f}% | peor {100 * v.min():+6.1f}%"


def main(meses=6):
    d = json.load(open("data/tendencia_acciones.json"))
    simbolos = sorted(d["cedear"].keys())
    datos = {s: preparar(x) for s, x in bajar(simbolos).items()}
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(months=meses)
    L = [f"CÍRCULOS VERDES EN CEDEARs, últimos {meses} meses ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Entrada: apertura de la rueda siguiente. Costo {100 * COSTO:.2f}% por lado. Precios en USD de la acción.", ""]
    ops, base = [], {k: [] for k in ["h5", "h10", "h20", "h40", "hoy", "pa", "pm"]}
    base_mes = {}
    for s, P in datos.items():
        f = P["f"]; n = len(f)
        for i in range(210, n - 1):
            if f[i] < ini: continue
            e = i + 1
            for k, hz in (("h5", 5), ("h10", 10), ("h20", 20), ("h40", 40)):
                r = fijo(P, e, hz)
                if r is not None:
                    base[k].append(r)
                    if k == "h20": base_mes.setdefault(f[i].strftime("%Y-%m"), []).append(r)
            base["hoy"].append(hasta_hoy(P, e))
            if not np.isnan(P["atrp"][i]):
                pa = plan(P, e, P["O"][e] * (1 - 2.5 * P["atrp"][i])); pm = plan(P, e, P["low20"][i])
                if pa: base["pa"].append(pa[0])
                if pm: base["pm"].append(pm[0])
            if P["verde"][i]:
                o = dict(s=s, f=f[i].date().isoformat(), strong=bool(P["strong"][i]), ent=float(P["O"][e]), hoy=hasta_hoy(P, e))
                for hz in HORIZ: o[f"h{hz}"] = fijo(P, e, hz)
                o["pa"] = plan(P, e, P["O"][e] * (1 - 2.5 * P["atrp"][i])); o["pm"] = plan(P, e, P["low20"][i])
                ops.append(o)
    pend = [s for s, P in datos.items() if P["verde"][-1] and P["f"][-1] >= ini]
    L.append(f"Señales verdes en la ventana: {len(ops)} ({sum(o['strong'] for o in ops)} Strong) en {len({o['s'] for o in ops})} acciones. Señales de la última rueda, todavía sin entrada: {', '.join(pend) or 'ninguna'}.")
    L.append("")
    for nom, g in [("TODAS LAS SEÑALES VERDES", ops), ("Solo Strong", [o for o in ops if o["strong"]]), ("Solo Verde (no Strong)", [o for o in ops if not o["strong"]])]:
        L.append(f"== {nom} ==")
        for hz in HORIZ: L.append(f"  Salir a {hz:>2} ruedas : " + resumen([o[f"h{hz}"] for o in g]))
        L.append("  Mantener hasta hoy  : " + resumen([o["hoy"] for o in g]))
        for k, t in (("pa", "Plan, stop por ATR (2,5×)"), ("pm", "Plan, stop amplio (mín. 20)")):
            rs = [o[k][1] for o in g if o[k]]
            L.append(f"  {t:<27}: " + resumen([o[k][0] if o[k] else None for o in g]) + (f" | R medio {np.mean(rs):+.2f}" if rs else ""))
        L.append("")
    L.append("== LÍNEA DE BASE: comprar cualquier acción con CEDEAR cualquier día de la misma ventana (mismas reglas) ==")
    for hz in HORIZ: L.append(f"  Salir a {hz:>2} ruedas : " + resumen(base[f"h{hz}"]))
    L.append("  Mantener hasta hoy  : " + resumen(base["hoy"]))
    L.append("  Plan, stop por ATR (2,5×)  : " + resumen(base["pa"])); L.append("  Plan, stop amplio (mín. 20): " + resumen(base["pm"]))
    L.append("")
    L.append("== MOTIVO DE SALIDA DEL PLAN (stop por ATR) ==")
    mot = pd.Series([o["pa"][2] for o in ops if o["pa"]]).value_counts()
    L.append("  " + " | ".join(f"{k}: {v}" for k, v in mot.items()))
    L.append("")
    L.append("== POR MES DE LA SEÑAL ==")
    for m in sorted({o["f"][:7] for o in ops}):
        g = [o for o in ops if o["f"][:7] == m]
        L.append(f"  {m}: {len(g):>3} señales | a 20 ruedas: " + corto([o["h20"] for o in g]) + " | hasta hoy: " + corto([o["hoy"] for o in g]) + " || base a 20 ruedas: " + corto(base_mes.get(m)))
    L.append("")
    L.append("== LISTA DE SEÑALES (fecha, acción, tipo, entrada, a 10 ruedas, a 20 ruedas, hasta hoy, plan ATR) ==")
    for o in sorted(ops, key=lambda x: x["f"]):
        g = lambda v: f"{100 * v:+6.1f}%" if v is not None else "     —"
        L.append(f"  {o['f']} {o['s']:<6} {'Strong' if o['strong'] else 'Verde ':<6} US${o['ent']:>8.2f} | {g(o['h10'])} | {g(o['h20'])} | {g(o['hoy'])} | {g(o['pa'][0]) if o['pa'] else '     —'} ({o['pa'][2] if o['pa'] else ''})")
    txt = "\n".join(L)
    open("data/investigacion_circulos.txt" if meses == 6 else f"data/investigacion_circulos_{meses}m.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
