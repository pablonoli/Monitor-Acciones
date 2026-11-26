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
    ent = P["O"][e]; R = ent - stop; obj = ent + 2 * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []; hk = None
    for j in range(e, min(n, e + I.MAXR)):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if lo <= sc:
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path, ((hk, vm) if mitad else None)
        if not mitad and hi >= obj:
            vm = max(obj, o) / ent * K; mitad = True; sc = ent; hk = j - e
        v = c / ent * K
        path.append(0.5 * vm + 0.5 * v if mitad else v)
    return path, ((hk, vm) if mitad else None)


def trade(P, i, spy_ok, camino=False):
    e = i + 1
    ent = P["O"][e]; stop = P["low20"][i]
    pl = I.plan(P, e, stop)
    if pl is None or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): return None
    ret = pl[0]; riesgo = (ent - stop) / ent
    flags = (1 if spy_ok[i] else 0) | (2 if P["sma50"][i] > P["sma200"][i] else 0) | (4 if P["atrp"][i] < 0.04 else 0)
    t = dict(f=P["f"][i], fe=P["f"][e], fx=P["f"][e + pl[3] - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=flags, anio=P["f"][i].year)
    if camino:
        t["path"], t["half"] = plan_path(P, e, stop); t["dts"] = list(P["f"][e:e + len(t["path"])])
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
        if hasattr(r, "path"): tomadas.append((eq * tam, tam, r.path, r.dts, r.half))
    cerrar(None)
    por, prev = {}, 1.0
    for a in sorted({e[0].year for e in eventos}):
        fin = [e[1] for e in eventos if e[0].year == a][-1]; por[a] = fin / prev - 1; prev = fin
    return dict(tom=n, acu=eq - 1, dd=dd, por=por, eventos=eventos, tomadas=tomadas)


def mercado(res, dias):
    """Capital día a día valuando las operaciones abiertas a precio de mercado. Devuelve (caída máx., prom. abiertas, prom. abiertas en días con posiciones, máx. abiertas, exposición prom., exposición máx., fecha del mínimo)."""
    ev = res["eventos"]; extra, cnt, expo = {}, {}, {}
    for usd, tam, path, dts, _ in res["tomadas"]:
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


def caja_datos(res, dias):
    """Números del flujo de caja de las operaciones tomadas (ver caja)."""
    tom, ev = res["tomadas"], res["eventos"]
    if not tom: return None
    dur = np.array([len(t[2]) for t in tom]); gan = np.array([t[2][-1] > 1 for t in tom]); mitad = np.array([t[4] is not None for t in tom])
    ent_mes = {}
    for usd, tam, path, dts, half in tom:
        if half: ent_mes[dts[half[0]].strftime("%Y-%m")] = ent_mes.get(dts[half[0]].strftime("%Y-%m"), 0.0) + usd * 0.5 * half[1]
        m = dts[-1].strftime("%Y-%m"); ent_mes[m] = ent_mes.get(m, 0.0) + usd * (path[-1] - (0.5 * half[1] if half else 0.0))
    meses = pd.period_range(dias[0], dias[-1], freq="M")
    def eq_en(d):
        v = 1.0
        for fx, e in ev:
            if fx <= d: v = e
            else: break
        return v
    inflow, pnl, cerr = [], [], []
    for mp in meses:
        ini_m, fin_m = mp.start_time - pd.Timedelta(days=1), mp.end_time
        e0, e1 = eq_en(ini_m), eq_en(fin_m)
        inflow.append(ent_mes.get(str(mp), 0.0) / e0); pnl.append(e1 / e0 - 1)
        cerr.append(sum(1 for t in tom if str(t[3][-1].to_period("M")) == str(mp)))
    inflow, pnl, cerr = np.array(inflow), np.array(pnl), np.array(cerr)
    racha_m = mx = 0
    for x in pnl:
        racha_m = racha_m + 1 if x <= 0 else 0; mx = max(mx, racha_m)
    cnt = {}
    for usd, tam, path, dts, half in tom:
        for d in dts[:-1]: cnt[d] = cnt.get(d, 0) + 1
    vacio = [cnt.get(d, 0) == 0 for d in dias]; rv = mv = 0
    for x in vacio:
        rv = rv + 1 if x else 0; mv = max(mv, rv)
    return dict(n=len(tom), dur=dur, gan=gan, mitad=mitad, inflow=inflow, pnl=pnl, cerr=cerr, racha_m=mx, vacio=np.mean(vacio), racha_v=mv, meses=len(meses))


def caja(res, dias):
    """Flujo de caja de las operaciones tomadas: cuánto tardan en cerrar, cuánto capital vuelve a efectivo por mes y cuántos meses seguidos pasan sin ganancia."""
    c = caja_datos(res, dias)
    if c is None: return ["sin operaciones"]
    dur, gan, mitad, inflow, pnl, cerr = c["dur"], c["gan"], c["mitad"], c["inflow"], c["pnl"], c["cerr"]
    return [f"  Operaciones tomadas: {c['n']}. Ruedas hasta cerrar: mediana {np.median(dur):.0f}, promedio {dur.mean():.1f}; cerradas en 5 ruedas o menos {100 * (dur <= 5).mean():.0f}%, en 10 o menos {100 * (dur <= 10).mean():.0f}%, en 20 o menos {100 * (dur <= 20).mean():.0f}%, llegaron al máximo de 60 ruedas {100 * (dur >= I.MAXR).mean():.0f}%.",
            f"  Duración promedio: ganadoras {dur[gan].mean():.1f} ruedas, perdedoras {dur[~gan].mean():.1f} ruedas. Vendieron la mitad (llegaron a +2R): {100 * mitad.mean():.0f}% de las operaciones.",
            f"  Capital que vuelve a efectivo por mes (% del capital; puede pasar de 100% porque la plata se vuelve a invertir): promedio {100 * inflow.mean():.0f}%, mediana {100 * np.median(inflow):.0f}%, mínimo {100 * inflow.min():.0f}%, máximo {100 * inflow.max():.0f}%.",
            f"  Ganancia realizada por mes (% del capital, {c['meses']} meses): promedio {100 * pnl.mean():+.2f}%, mediana {100 * np.median(pnl):+.2f}%, peor {100 * pnl.min():+.1f}%, mejor {100 * pnl.max():+.1f}%; meses positivos {100 * (pnl > 0).mean():.0f}%, negativos {100 * (pnl < 0).mean():.0f}%.",
            f"  Meses seguidos sin ganancia (≤ 0): máximo {c['racha_m']}. Meses sin ninguna operación cerrada: {int((cerr == 0).sum())}. Días de rueda sin ninguna operación abierta: {100 * c['vacio']:.0f}% (la racha más larga fue de {c['racha_v']} ruedas)."]


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
    sig, base, cruz = [], [], []
    for s, P in datos.items():
        ok = spy_ok_s.reindex(P["f"], method="ffill").fillna(False).values
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini: continue
            es_cruce = bool(P["sma50"][i] > P["sma200"][i] and P["sma50"][i - 1] <= P["sma200"][i - 1])   # la SMA 50 cruza por encima de la 200 en el cierre de i
            t = trade(P, i, ok, camino=bool(P["verde"][i]) or es_cruce)
            if t is None: continue
            t["s"] = s
            if P["verde"][i]: sig.append(dict(t, strong=bool(P["strong"][i])))
            if es_cruce: cruz.append(dict(t))
            base.append({k: v for k, v in t.items() if k not in ("path", "dts", "half")})
    S, B, C = pd.DataFrame(sig), pd.DataFrame(base), pd.DataFrame(cruz)
    # versiones: (nombre, DataFrame, máscara de filtros)
    VER = [(nom(m), S, m) for m in [0, 1, 2, 4, 3, 5, 6, 7]] + [("CRUCE DORADO (sin círculo verde)", C, 0), ("CRUCE DORADO + SPY>SMA200", C, 1)]
    L = [f"CÍRCULOS VERDES EN CEDEARs CON FILTROS, últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Plan con stop amplio (mín. 20 ruedas), mitad a +2R, stop a la entrada, máx. 60 ruedas. Costo {100 * I.COSTO:.2f}% por lado.",
         "Límites: la lista de acciones es la de hoy (hay sesgo hacia las que sobrevivieron y subieron); en prom./mediana/R cada operación cuenta igual.",
         "R = resultado neto / distancia al stop. Capital: cada operación arriesga 1% del capital (tamaño = 1% / distancia al stop), sin apalancamiento (exposición total máxima 100%: si no hay lugar, se saltea la señal; 'toma.' = operaciones realmente tomadas), el capital se actualiza al cerrar cada operación.",
         "Caída máx. = solo con operaciones cerradas. Caída MTM = valuando además las operaciones abiertas a precio de cierre de cada día.", ""]
    enc = f"{'Versión':<44}| {'ops':>5} | {'prom.':>7} | {'mediana':>7} | {'gan.':>4} | {'R med':>5} | {'racha':>4} | {'toma.':>5} | {'acumulado':>9} | {'caída máx':>9} | {'caída MTM':>9}"
    mascaras = [0, 1, 2, 4, 3, 5, 6, 7]
    sb_acu, sb_dd, sb_por = spy_bh(ini, ult, ini_dl)
    L += ["== SEÑALES VERDES CON CADA FILTRO (el plan con 1% de riesgo por operación) ==", enc]
    res = {}
    for nombre, D, m in VER:
        sub = D[(D["fl"] & m) == m]; res[nombre] = sim(sub)
        if nombre.startswith("CRUCE") and nombre.endswith("(sin círculo verde)"): L.append("-- entrada el día después del cruce de la SMA 50 sobre la SMA 200 (sin círculo verde), mismo plan --")
        L.append(f"{nombre:<44}| " + fila(sub, dias))
    L.append(f"{'SPY comprar y mantener (retorno total)':<44}|     — |       — |       — |    — |     — |    — |     — | {100 * sb_acu:+8.1f}% | {100 * sb_dd:+6.1f}% | {100 * sb_dd:+7.1f}%")
    L += ["", "== LÍNEA DE BASE: cualquier acción, cualquier día, mismo plan y mismo filtro (sin círculo verde) ==", enc.split("| racha")[0]]
    for m in mascaras:
        sub = B[(B["fl"] & m) == m]
        L.append(f"{nom(m):<44}| " + fila(sub))
    L += ["", "== VENTAJA = promedio de la señal − promedio de la base con el mismo filtro (puntos porcentuales) =="]
    for m in mascaras:
        a, b = S[(S["fl"] & m) == m]["ret"], B[(B["fl"] & m) == m]["ret"]
        L.append(f"{nom(m):<44}| {100 * (a.mean() - b.mean()):+6.2f} pp  (señal {100 * a.mean():+.2f}% vs base {100 * b.mean():+.2f}%)" if len(a) else f"{nom(m):<44}| sin operaciones")
    for nombre, m_base in (("CRUCE DORADO (sin círculo verde)", 2), ("CRUCE DORADO + SPY>SMA200", 3)):
        a = C[(C["fl"] & (1 if "SPY" in nombre else 0)) == (1 if "SPY" in nombre else 0)]["ret"]; b = B[(B["fl"] & m_base) == m_base]["ret"]
        L.append(f"{nombre:<44}| {100 * (a.mean() - b.mean()):+6.2f} pp  (cruce {100 * a.mean():+.2f}% vs base {nom(m_base)} {100 * b.mean():+.2f}%)")
    anios_l = sorted(set(S["anio"].unique()) | set(C["anio"].unique()))
    L += ["", "== POR AÑO DE LA SEÑAL: cantidad de operaciones y promedio por operación (señal | base con el mismo filtro) =="]
    for m in mascaras:
        L.append(f"-- {nom(m)}")
        for a in anios_l:
            s1, b1 = S[(S["anio"] == a) & ((S["fl"] & m) == m)], B[(B["anio"] == a) & ((B["fl"] & m) == m)]
            L.append(f"   {a}{' (parcial)' if a in (ini.year, ult.year) else '':<12}señal {len(s1):>4} ops {100 * s1['ret'].mean() if len(s1) else float('nan'):+7.2f}%, gan. {100 * (s1['ret'] > 0).mean() if len(s1) else float('nan'):3.0f}%, R {s1['rnet'].mean() if len(s1) else float('nan'):+5.2f} | base {len(b1):>6} ops {100 * b1['ret'].mean() if len(b1) else float('nan'):+7.2f}%")
    for nombre, D, m in VER[8:]:
        L.append(f"-- {nombre}")
        for a in anios_l:
            s1 = D[(D["anio"] == a) & ((D["fl"] & m) == m)]
            L.append(f"   {a}{' (parcial)' if a in (ini.year, ult.year) else '':<12}cruce {len(s1):>4} ops {100 * s1['ret'].mean() if len(s1) else float('nan'):+7.2f}%, gan. {100 * (s1['ret'] > 0).mean() if len(s1) else float('nan'):3.0f}%, R {s1['rnet'].mean() if len(s1) else float('nan'):+5.2f}")
    L += ["", "== RETORNO DEL CAPITAL POR AÑO (1% de riesgo por operación, sin apalancamiento, por año de cierre de las operaciones; SPY: retorno total del año calendario) =="]
    ya = sorted(set(anios_l) | set(sb_por))
    L.append(f"{'Versión':<44}" + "".join(f"{a:>10}" for a in ya))
    for nombre, D, m in VER:
        L.append(f"{nombre:<44}" + "".join(f"{100 * res[nombre]['por'][a]:+9.1f}%" if a in res[nombre]["por"] else f"{'—':>10}" for a in ya))
    L.append(f"{'SPY comprar y mantener':<44}" + "".join(f"{100 * sb_por[a]:+9.1f}%" if a in sb_por else f"{'—':>10}" for a in ya))
    L += ["", f"== OPERACIONES ABIERTAS A LA VEZ y EXPOSICIÓN (días de rueda de la ventana: {len(dias)}) =="]
    L.append(f"{'Versión':<44}| {'abiertas prom.':>14} | {'prom. con pos.':>14} | {'máx.':>4} | {'exposición prom.':>16} | {'exposición máx.':>15} | {'peor caída MTM el':>17}")
    for nombre, D, m in VER:
        mm = mercado(res[nombre], dias)
        L.append(f"{nombre:<44}| {mm[1]:>14.1f} | {mm[2]:>14.1f} | {mm[3]:>4} | {100 * mm[4]:>15.0f}% | {100 * mm[5]:>14.0f}% | {str(mm[6].date()) if mm[6] is not None else '—':>17}")
    L += ["", "== FLUJO DE CAJA (qué tan rápido vuelve la plata a efectivo) =="]
    for nombre in ["Sin filtro", "SPY>SMA200 + SMA50>SMA200", "SPY>SMA200 + SMA50>SMA200 + ATR%<4%", "CRUCE DORADO + SPY>SMA200"]:
        L += [f"-- {nombre}"] + caja(res[nombre], dias)
    txt = "\n".join(L); open("data/investigacion_circulos_filtros.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
