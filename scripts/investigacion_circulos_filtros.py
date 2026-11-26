"""Investigación: círculos verdes en CEDEARs con filtros (SPY sobre su SMA 200, SMA 50 > SMA 200 de la acción, ATR % < 4%) y por año.
Usa el plan con stop amplio (mínimo de 20 ruedas): vende la mitad a +2× el riesgo, mueve el stop a la entrada, máximo 60 ruedas. Entrada en la apertura siguiente, costo 0,15% por lado.
Compara con la línea de base: comprar cualquier acción con CEDEAR cualquier día de la ventana con el mismo plan (sin filtro y con el mismo filtro) y con comprar y mantener SPY.
Capital: cada operación arriesga 1% (tamaño = 1% / distancia al stop), sin apalancamiento (exposición máx. 100%). Caída máxima también valuando las operaciones abiertas a precio de mercado cada día.
Ventana: 5 años (argumento opcional: años). Resultado: data/investigacion_circulos_filtros.txt"""
import json, sys, heapq
import numpy as np, pandas as pd, yfinance as yf
import investigacion_circulos as I

NOMBRES = {1: "SPY>SMA200", 2: "SMA50>SMA200", 4: "ATR%<4%"}
K = (1 - I.COSTO) / (1 + I.COSTO)


def nom(m):
    return "Sin filtro" if m == 0 else " + ".join(NOMBRES[b] for b in (1, 2, 4) if m & b)


def plan_path(P, e, stop):
    """Igual que I.plan pero devuelve el valor de la operación (por unidad invertida, neto de costos) al cierre de cada rueda; el último valor es el realizado."""
    ent = P["O"][e]; R = ent - stop; obj = ent + 2 * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []
    for j in range(e, min(n, e + I.MAXR)):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if lo <= sc:
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path
        if not mitad and hi >= obj:
            vm = max(obj, o) / ent * K; mitad = True; sc = ent
        v = c / ent * K
        path.append(0.5 * vm + 0.5 * v if mitad else v)
    return path


def trade(P, i, spy_ok, camino=False):
    e = i + 1
    ent = P["O"][e]; stop = P["low20"][i]
    pl = I.plan(P, e, stop)
    if pl is None or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): return None
    ret = pl[0]; riesgo = (ent - stop) / ent
    flags = (1 if spy_ok[i] else 0) | (2 if P["sma50"][i] > P["sma200"][i] else 0) | (4 if P["atrp"][i] < 0.04 else 0)
    t = dict(f=P["f"][i], fe=P["f"][e], fx=P["f"][e + pl[3] - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=flags, anio=P["f"][i].year)
    if camino:
        t["path"] = plan_path(P, e, stop); t["dts"] = list(P["f"][e:e + len(t["path"])])
        assert len(t["path"]) == pl[3] and abs(t["path"][-1] - 1 - ret) < 1e-9, (t["path"][-1] - 1, ret)
    return t


def racha(v):
    mx = a = 0
    for x in v:
        a = a + 1 if x < 0 else 0; mx = max(mx, a)
    return mx


def sim(df, riesgo_op=0.01, tope=1.0):
    """Capital sin apalancamiento: cada operación arriesga 1% del capital (tamaño = 1% / distancia al stop), con tope de exposición total del 100%; si no hay lugar, se saltea.
    El capital realizado se actualiza al cerrar cada operación. Devuelve dict con operaciones tomadas, acumulado, caída máx. (solo cerradas), retorno por año y los datos para valuar a mercado."""
    df = df.sort_values(["fe", "s"]); eq, expo, ab = 1.0, 0.0, []
    eventos, tomadas = [], []; pico = 1.0; dd = 0.0; n = 0
    def cerrar(hasta):
        nonlocal eq, expo, pico, dd
        while ab and (hasta is None or ab[0][0] <= hasta):
            fx, _, tam, pnl = heapq.heappop(ab); eq += pnl; expo -= tam; pico = max(pico, eq); dd = min(dd, eq / pico - 1); eventos.append((fx, eq))
    for r in df.itertuples():
        cerrar(r.fe)
        tam = min(riesgo_op / r.riesgo, tope - expo)
        if tam < 0.02: continue
        n += 1; expo += tam; heapq.heappush(ab, (r.fx, n, tam, eq * tam * r.ret))
        if hasattr(r, "path"): tomadas.append((eq * tam, tam, r.path, r.dts))
    cerrar(None)
    por, prev = {}, 1.0
    for a in sorted({e[0].year for e in eventos}):
        fin = [e[1] for e in eventos if e[0].year == a][-1]; por[a] = fin / prev - 1; prev = fin
    return dict(tom=n, acu=eq - 1, dd=dd, por=por, eventos=eventos, tomadas=tomadas)


def mercado(res, dias):
    """Capital día a día valuando las operaciones abiertas a precio de mercado. Devuelve (caída máx., prom. abiertas, prom. abiertas en días con posiciones, máx. abiertas, exposición prom., exposición máx., fecha del mínimo)."""
    ev = res["eventos"]; extra, cnt, expo = {}, {}, {}
    for usd, tam, path, dts in res["tomadas"]:
        for k in range(len(dts) - 1):                    # el último día de la operación ya está realizado
            d = dts[k]; extra[d] = extra.get(d, 0.0) + usd * (path[k] - 1); cnt[d] = cnt.get(d, 0) + 1; expo[d] = expo.get(d, 0.0) + tam
    j, real, pico, dd, dmin, ser = 0, 1.0, 1.0, 0.0, None, []
    for d in dias:
        while j < len(ev) and ev[j][0] <= d: real = ev[j][1]; j += 1
        v = real + extra.get(d, 0.0); pico = max(pico, v)
        if v / pico - 1 < dd: dd, dmin = v / pico - 1, d
        ser.append(cnt.get(d, 0))
    ser = np.array(ser); ex = np.array([expo.get(d, 0.0) for d in dias])
    return dd, ser.mean(), ser[ser > 0].mean() if (ser > 0).any() else 0.0, int(ser.max()), ex.mean(), ex.max(), dmin


def fila(df, dias=None):
    if not len(df): return "   0 ops"
    r, R = df["ret"].values, df["rnet"].values
    t = f"{len(df):>5} | {100 * r.mean():+6.2f}% | {100 * np.median(r):+6.2f}% | {100 * (r > 0).mean():4.0f}% | {R.mean():+5.2f}"
    if dias is not None:
        res = sim(df); m = mercado(res, dias)
        t += f" | {racha(df.sort_values(['fe', 's'])['ret'].values):>4} | {res['tom']:>5} | {100 * res['acu']:+8.1f}% | {100 * res['dd']:+6.1f}% | {100 * m[0]:+7.1f}%"
    return t


def spy_bh(ini, ult, ini_dl):
    x = yf.download("SPY", start=ini_dl, interval="1d", auto_adjust=True, progress=False)["Close"]
    x = x.squeeze().dropna(); x = x[(x.index >= ini) & (x.index <= ult)]
    pico = x.cummax(); dd = (x / pico - 1).min(); por, prev = {}, x.iloc[0]
    for a in sorted({d.year for d in x.index}):
        fin = x[[d.year == a for d in x.index]].iloc[-1]; por[a] = fin / prev - 1; prev = fin
    return x.iloc[-1] / x.iloc[0] - 1, dd, por


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]; spy_ok_s = (sx["Close"] > sx["Close"].rolling(200).mean())
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios)
    dias = [x for x in sx.index if ini <= x <= ult]
    sig, base = [], []
    for s, P in datos.items():
        ok = spy_ok_s.reindex(P["f"], method="ffill").fillna(False).values
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini: continue
            t = trade(P, i, ok, camino=bool(P["verde"][i]))
            if t is None: continue
            t["s"] = s
            if P["verde"][i]: sig.append(dict(t, strong=bool(P["strong"][i])))
            base.append({k: v for k, v in t.items() if k not in ("path", "dts")})
    S, B = pd.DataFrame(sig), pd.DataFrame(base)
    L = [f"CÍRCULOS VERDES EN CEDEARs CON FILTROS, últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Plan con stop amplio (mín. 20 ruedas), mitad a +2R, stop a la entrada, máx. 60 ruedas. Costo {100 * I.COSTO:.2f}% por lado.",
         "Límites: la lista de acciones es la de hoy (hay sesgo hacia las que sobrevivieron y subieron); en prom./mediana/R cada operación cuenta igual.",
         "R = resultado neto / distancia al stop. Capital: cada operación arriesga 1% del capital (tamaño = 1% / distancia al stop), sin apalancamiento (exposición total máxima 100%: si no hay lugar, se saltea la señal; 'toma.' = operaciones realmente tomadas), el capital se actualiza al cerrar cada operación.",
         "Caída máx. = solo con operaciones cerradas. Caída MTM = valuando además las operaciones abiertas a precio de cierre de cada día.", ""]
    enc = f"{'Versión':<44}| {'ops':>5} | {'prom.':>7} | {'mediana':>7} | {'gan.':>4} | {'R med':>5} | {'racha':>4} | {'toma.':>5} | {'acumulado':>9} | {'caída máx':>9} | {'caída MTM':>9}"
    mascaras = [0, 1, 2, 4, 3, 5, 6, 7]
    sb_acu, sb_dd, sb_por = spy_bh(ini, ult, ini_dl)
    L += ["== SEÑALES VERDES CON CADA FILTRO (el plan con 1% de riesgo por operación) ==", enc]
    res = {}
    for m in mascaras:
        sub = S[(S["fl"] & m) == m]; res[m] = sim(sub)
        L.append(f"{nom(m):<44}| " + fila(sub, dias))
    L.append(f"{'SPY comprar y mantener (retorno total)':<44}|     — |       — |       — |    — |     — |    — |     — | {100 * sb_acu:+8.1f}% | {100 * sb_dd:+6.1f}% | {100 * sb_dd:+7.1f}%")
    L += ["", "== LÍNEA DE BASE: cualquier acción, cualquier día, mismo plan y mismo filtro (sin círculo verde) ==", enc.split("| racha")[0]]
    for m in mascaras:
        sub = B[(B["fl"] & m) == m]
        L.append(f"{nom(m):<44}| " + fila(sub))
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
    L += ["", "== RETORNO DEL CAPITAL POR AÑO (1% de riesgo por operación, sin apalancamiento, por año de cierre de las operaciones; SPY: retorno total del año calendario) =="]
    ya = sorted(set(anios_l) | set(sb_por))
    L.append(f"{'Versión':<44}" + "".join(f"{a:>10}" for a in ya))
    for m in mascaras:
        L.append(f"{nom(m):<44}" + "".join(f"{100 * res[m]['por'][a]:+9.1f}%" if a in res[m]["por"] else f"{'—':>10}" for a in ya))
    L.append(f"{'SPY comprar y mantener':<44}" + "".join(f"{100 * sb_por[a]:+9.1f}%" if a in sb_por else f"{'—':>10}" for a in ya))
    L += ["", f"== OPERACIONES ABIERTAS A LA VEZ y EXPOSICIÓN (días de rueda de la ventana: {len(dias)}) =="]
    L.append(f"{'Versión':<44}| {'abiertas prom.':>14} | {'prom. con pos.':>14} | {'máx.':>4} | {'exposición prom.':>16} | {'exposición máx.':>15} | {'peor caída MTM el':>17}")
    for m in mascaras:
        mm = mercado(res[m], dias)
        L.append(f"{nom(m):<44}| {mm[1]:>14.1f} | {mm[2]:>14.1f} | {mm[3]:>4} | {100 * mm[4]:>15.0f}% | {100 * mm[5]:>14.0f}% | {mm[6].date() if mm[6] is not None else '—':>17}")
    txt = "\n".join(L); open("data/investigacion_circulos_filtros.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
