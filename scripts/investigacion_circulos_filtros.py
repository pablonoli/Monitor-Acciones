"""Investigación: círculos verdes en CEDEARs con filtros (SPY sobre su SMA 200, SMA 50 > SMA 200 de la acción, ATR % < 4%) y por año.
Usa el plan con stop amplio (mínimo de 20 ruedas): vende la mitad a +2× el riesgo, mueve el stop a la entrada, máximo 60 ruedas. Entrada en la apertura siguiente, costo 0,15% por lado.
Compara con la línea de base: comprar cualquier acción con CEDEAR cualquier día de la ventana con el mismo plan (sin filtro y con el mismo filtro).
Ventana: 5 años (argumento opcional: años). Resultado: data/investigacion_circulos_filtros.txt"""
import json, sys, itertools
import numpy as np, pandas as pd
import investigacion_circulos as I

NOMBRES = {0: "Sin filtro", 1: "SPY>SMA200", 2: "SMA50>SMA200", 4: "ATR%<4%"}


def nom(m):
    return "Sin filtro" if m == 0 else " + ".join(NOMBRES[b] for b in (1, 2, 4) if m & b)


def trade(P, i, spy_ok):
    e = i + 1
    ent = P["O"][e]; stop = P["low20"][i]
    pl = I.plan(P, e, stop)
    if pl is None or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): return None
    ret = pl[0]; riesgo = (ent - stop) / ent
    flags = (1 if spy_ok[i] else 0) | (2 if P["sma50"][i] > P["sma200"][i] else 0) | (4 if P["atrp"][i] < 0.04 else 0)
    return dict(f=P["f"][i], fe=P["f"][e], ret=ret, rnet=ret / riesgo, fl=flags, anio=P["f"][i].year)


def racha(v):
    mx = a = 0
    for x in v:
        a = a + 1 if x < 0 else 0; mx = max(mx, a)
    return mx


def fila(df, capital=True):
    if not len(df): return "   0 ops"
    df = df.sort_values(["fe", "s"]) if "s" in df else df.sort_values("fe")
    r, R = df["ret"].values, df["rnet"].values
    t = f"{len(df):>5} | {100 * r.mean():+6.2f}% | {100 * np.median(r):+6.2f}% | {100 * (r > 0).mean():4.0f}% | {R.mean():+5.2f}"
    if capital:
        eq = np.cumprod(1 + 0.01 * R); pico = np.maximum.accumulate(np.r_[1, eq])[1:]
        t += f" | {racha(r):>4} | {100 * (eq[-1] - 1):+7.1f}% | {100 * ((eq / pico - 1).min()):+6.1f}%"
    return t


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]; spy_ok_s = (sx["Close"] > sx["Close"].rolling(200).mean())
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios)
    sig, base = [], []
    for s, P in datos.items():
        ok = spy_ok_s.reindex(P["f"], method="ffill").fillna(False).values
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini: continue
            t = trade(P, i, ok)
            if t is None: continue
            t["s"] = s; base.append(t)
            if P["verde"][i]: sig.append(dict(t, strong=bool(P["strong"][i])))
    S, B = pd.DataFrame(sig), pd.DataFrame(base)
    L = [f"CÍRCULOS VERDES EN CEDEARs CON FILTROS, últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Plan con stop amplio (mín. 20 ruedas), mitad a +2R, stop a la entrada, máx. 60 ruedas. Costo {100 * I.COSTO:.2f}% por lado.",
         "Límites: la lista de acciones es la de hoy (hay sesgo hacia las que sobrevivieron y subieron); las operaciones se ordenan por fecha de entrada y se suponen independientes (en la realidad se superponen).",
         "R = resultado neto / distancia al stop. Capital: cada operación arriesga 1% del capital actual (resultado = 1% × R), compuesto en orden de entrada.", ""]
    enc = f"{'Versión':<44}| {'ops':>5} | {'prom.':>7} | {'mediana':>7} | {'gan.':>4} | {'R med':>5} | {'racha':>4} | {'acumulado':>8} | {'caída máx':>9}"
    L += ["== SEÑALES VERDES CON CADA FILTRO ==", enc]
    mascaras = [0, 1, 2, 4, 3, 5, 6, 7]
    for m in mascaras:
        sub = S[(S["fl"] & m) == m]
        L.append(f"{nom(m):<44}| " + fila(sub))
    L += ["", "== LÍNEA DE BASE: cualquier acción, cualquier día, mismo plan y mismo filtro (sin círculo verde) ==", enc.replace("| racha | acumulado | caída máx", "")]
    for m in mascaras:
        sub = B[(B["fl"] & m) == m]
        L.append(f"{nom(m):<44}| " + fila(sub, capital=False))
    L += ["", "== VENTAJA = promedio de la señal − promedio de la base con el mismo filtro (puntos porcentuales) =="]
    for m in mascaras:
        a, b = S[(S["fl"] & m) == m]["ret"], B[(B["fl"] & m) == m]["ret"]
        L.append(f"{nom(m):<44}| {100 * (a.mean() - b.mean()):+6.2f} pp  (señal {100 * a.mean():+.2f}% vs base {100 * b.mean():+.2f}%)" if len(a) else f"{nom(m):<44}| sin operaciones")
    anios_l = sorted(S["anio"].unique())
    L += ["", "== POR AÑO DE LA SEÑAL: cantidad de operaciones y promedio por operación (señal | base con el mismo filtro) =="]
    for m in mascaras:
        L.append(f"-- {nom(m)}")
        for a in anios_l:
            s1, b1 = S[(S["anio"] == a) & ((S["fl"] & m) == m)], B[(B["anio"] == a) & ((B["fl"] & m) == m)]
            L.append(f"   {a}{' (parcial)' if a in (ini.year, ult.year) else '':<12}señal {len(s1):>4} ops {100 * s1['ret'].mean() if len(s1) else float('nan'):+7.2f}%, gan. {100 * (s1['ret'] > 0).mean() if len(s1) else float('nan'):3.0f}%, R {s1['rnet'].mean() if len(s1) else float('nan'):+5.2f} | base {len(b1):>6} ops {100 * b1['ret'].mean() if len(b1) else float('nan'):+7.2f}%")
    L += ["", "== ACUMULADO POR AÑO (1% de riesgo por operación) =="]
    L.append(f"{'Versión':<44}" + "".join(f"{a:>10}" for a in anios_l))
    for m in mascaras:
        cel = []
        for a in anios_l:
            s1 = S[(S["anio"] == a) & ((S["fl"] & m) == m)].sort_values(["fe", "s"])
            cel.append(f"{100 * (np.prod(1 + 0.01 * s1['rnet'].values) - 1):+9.1f}%" if len(s1) else f"{'—':>10}")
        L.append(f"{nom(m):<44}" + "".join(cel))
    txt = "\n".join(L); open("data/investigacion_circulos_filtros.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
