"""Investigación: en BTC, SOL y HYPE, ¿qué rinde más: la estrategia actual del dashboard (cruce de medias exponenciales 10/30 diario con filtro
"BTC o propio" de 200 días) o el círculo verde con el plan de las acciones? Últimos 5 años (HYPE desde que cotiza).

Cruce 10/30 (igual que el dashboard): comprado mientras la EMA 10 está arriba de la EMA 30 y hay permiso (BTC arriba de su SMA 200 o el propio activo
arriba de la suya); la señal sale al cierre y se ejecuta en la apertura siguiente. Costo 0,15% por compra y por venta (como el dashboard).
Círculo verde (igual que las acciones): techo = máximo de los 20 días anteriores; ruptura = cierre > techo, cierre de ayer <= techo de ayer y Z > 1,5;
verde = ruptura con cierre sobre la SMA 200; Strong = Z > 2,5 y volumen > 1,5 x promedio de 20 días. Plan: compra en la apertura siguiente, stop en el
mínimo de 20 días, mitad a +2R y stop al precio de compra, cierre de todo al cierre del día 20 (cripto opera todos los días: unas 3 semanas).
Una operación a la vez por moneda (las señales mientras hay una abierta se ignoran).
Dos formas de usar el capital:
 - "porción": cada moneda tiene 1/3 del capital y cada operación usa toda su porción (como el dashboard de cripto).
 - "riesgo 1%": capital común, cada operación arriesga 1% del capital (tamaño = 1% / distancia al stop), máximo 1/3 del capital por moneda.
Resultado: data/investigacion_cripto_circulos.txt"""
import sys
import numpy as np, pandas as pd, yfinance as yf

COSTO = 0.0015
K = (1 - COSTO) / (1 + COSTO)
MONEDAS = [("BTC", "BTC-USD"), ("SOL", "SOL-USD"), ("HYPE", "HYPE32196-USD")]
CORTE = pd.Timestamp("2024-05-26")


def bajar(t):
    for intento in range(3):
        try:
            x = yf.download(t, start="2019-01-01", interval="1d", auto_adjust=False, progress=False)
            if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
            x = x[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
            if len(x) > 60: x.index = pd.to_datetime(x.index).tz_localize(None); return x
        except Exception as e:
            print(t, "falló", e, file=sys.stderr)
    return None


def ops_cruce(x, btc_ok):
    c = x["Close"]; f = c.ewm(span=10, adjust=False).mean(); s = c.ewm(span=30, adjust=False).mean()
    propio = c > c.rolling(200).mean()
    perm = propio | btc_ok.reindex(x.index).fillna(False)
    cond = ((f > s) & perm & (np.arange(len(c)) >= 30)).values
    O, C, idx = x["Open"].values, c.values, x.index; out = []; e = None
    for i in range(len(c) - 1):
        if e is None and cond[i]: e = i + 1
        elif e is not None and not cond[i]:
            j = i + 1; path = list(C[e:j] / O[e] * K) + [O[j] / O[e] * K * K]
            out.append(dict(e=idx[e], dts=list(idx[e:j + 1]), path=path, riesgo=None)); e = None
    if e is not None:
        out.append(dict(e=idx[e], dts=list(idx[e:]), path=list(C[e:] / O[e] * K), riesgo=None, abierta=True))
    return out


def ops_circulo(x, solo_regla):
    o, h, l, c, v = (x[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    techo = h.shift(1).rolling(20).max(); mom = c.diff(); z = (mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)
    verde = (c > techo) & (c.shift(1) <= techo.shift(1)) & (z > 1.5) & (c > c.rolling(200).mean())
    strong = verde & (z > 2.5) & (v > 1.5 * v.rolling(20).mean())
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1); atrp = (tr / c).rolling(14).mean()
    sen = (strong & (atrp >= 0.02)) if solo_regla else verde
    low20 = l.rolling(20).min()
    O, H, L, C, idx, n = o.values, h.values, l.values, c.values, x.index, len(x); out = []; libre = 0
    for i in range(n - 1):
        if i < libre or not bool(sen.iloc[i]): continue
        e = i + 1; ent = O[e]; st = float(low20.iloc[i])
        if not (0 < st < ent): continue
        R = ent - st; obj = ent + 2 * R; sc = st; mitad = None; path = []; j = e
        while j < min(n, e + 20):
            if L[j] <= sc:
                vv = min(sc, O[j]) / ent * K; path.append(0.5 * mitad + 0.5 * vv if mitad else vv); break
            if mitad is None and H[j] >= obj: mitad = max(obj, O[j]) / ent * K; sc = ent
            path.append(0.5 * mitad + 0.5 * C[j] / ent * K if mitad else C[j] / ent * K); j += 1
        fin = e + len(path) - 1
        out.append(dict(e=idx[e], dts=list(idx[e:fin + 1]), path=path, riesgo=(ent - st) / ent, abierta=(fin == n - 1 and len(path) < 20)))
        libre = fin + 1   # una operación a la vez: la próxima señal tiene que ser después de la salida
    return out


def curva_porcion(ops, dias, desde):
    """Equity diaria de una moneda usando toda su porción en cada operación."""
    eq = pd.Series(1.0, index=dias); nivel = 1.0; marca = {}
    for op in ops:
        if op["e"] < desde: continue
        for d, val in zip(op["dts"], op["path"]): marca[d] = nivel * val
        nivel *= op["path"][-1]; marca[op["dts"][-1]] = nivel
    ult = 1.0
    for d in dias:
        if d in marca: ult = marca[d]
        eq[d] = ult
    # entre operaciones queda el último nivel cerrado
    return eq


def sim_riesgo(opsd, dias, desde):
    """Capital común, 1% de riesgo por operación, máximo 1/3 por moneda y 100% en total. Devuelve equity diaria (valuada a mercado)."""
    eventos = sorted([(op["e"], m, op) for m, L in opsd.items() for op in L if op["e"] >= desde], key=lambda t: t[0])
    caja = 1.0; abiertas = []; eq = []; k = 0
    for d in dias:
        while k < len(eventos) and eventos[k][0] <= d:
            _, m, op = eventos[k]; k += 1
            if any(a["m"] == m for a in abiertas): continue
            total = caja + sum(a["monto"] * a["val"] for a in abiertas)
            usado = sum(a["monto"] for a in abiertas)
            tam = min(0.01 / op["riesgo"], 1 / 3) * total if op["riesgo"] else total / 3
            tam = min(tam, caja)
            if tam <= 0: continue
            caja -= tam; abiertas.append(dict(m=m, op=op, monto=tam, val=1.0, pos={dd: v for dd, v in zip(op["dts"], op["path"])}, fin=op["dts"][-1]))
        for a in abiertas:
            if d in a["pos"]: a["val"] = a["pos"][d]
        cerr = [a for a in abiertas if a["fin"] <= d and not a["op"].get("abierta")]
        for a in cerr: caja += a["monto"] * a["val"]; abiertas.remove(a)
        eq.append(caja + sum(a["monto"] * a["val"] for a in abiertas))
    return pd.Series(eq, index=dias)


def met(eq):
    dd = (eq / eq.cummax() - 1).min(); return eq.iloc[-1] - 1, dd


def stats(ops, desde, hasta=None):
    r = [op["path"][-1] - 1 for op in ops if op["e"] >= desde and (hasta is None or op["e"] < hasta)]
    if not r: return "sin operaciones"
    r = np.array(r); return f"{len(r):>3} op. · {100 * (r > 0).mean():3.0f}% ganadoras · promedio {100 * r.mean():+6.1f}% · mediana {100 * np.median(r):+6.1f}%"


def main():
    datos = {m: bajar(t) for m, t in MONEDAS}
    datos = {m: x for m, x in datos.items() if x is not None}
    btc = datos["BTC"]; btc_ok = btc["Close"] > btc["Close"].rolling(200).mean()
    ult = max(x.index[-1] for x in datos.values()); ini = ult - pd.DateOffset(years=5)
    dias = pd.date_range(ini, ult, freq="D")
    EST = {"Cruce 10/30 (el actual)": {m: ops_cruce(x, btc_ok) for m, x in datos.items()},
           "Círculo verde (todos)": {m: ops_circulo(x, False) for m, x in datos.items()},
           "Tu regla (Strong + ATR 2%)": {m: ops_circulo(x, True) for m, x in datos.items()}}
    L = [f"CRIPTO: CRUCE 10/30 CONTRA CÍRCULO VERDE. {ini.date()} a {ult.date()}. Monedas: {', '.join(f'{m} (desde {x.index[0].date()})' for m, x in datos.items())}.",
         "Costo 0,15% por compra y por venta. Una operación a la vez por moneda. Plan del círculo: stop en el mínimo de 20 días, mitad en +2R, cierre en el día 20.",
         "LÍMITES: solo 3 monedas (HYPE con muy poca historia): pocas operaciones, el resultado puede ser casualidad; lista elegida hoy; 3 estrategias probadas.", ""]
    L.append("== POR MONEDA (cada una con todo su capital en cada operación) ==")
    for m, x in datos.items():
        bh = x["Close"][x.index >= ini]; bhr = bh.iloc[-1] / bh.iloc[0] - 1; bdd = (bh / bh.cummax() - 1).min()
        L.append(f"{m}: comprar y mantener {100 * bhr:+.0f}% (caída máx. {100 * bdd:.0f}%)")
        dm = pd.date_range(max(ini, x.index[0]), ult, freq="D")
        for nom, opsd in EST.items():
            a, dd = met(curva_porcion(opsd[m], dm, ini))
            L.append(f"   {nom:<28} acumulado {100 * a:+7.0f}% · caída máx. {100 * dd:5.0f}% · {stats(opsd[m], ini)}")
    L += ["", "== LAS TRES JUNTAS (porción: 1/3 del capital por moneda; HYPE en efectivo hasta que existe) =="]
    bhs = []
    for m, x in datos.items():
        cc = x["Close"].reindex(dias).ffill(); cc = cc / cc.dropna().iloc[0]; bhs.append(cc.fillna(1.0))
    bh = sum(bhs) / len(bhs); a, dd = met(bh); L.append(f"   {'Comprar y mantener las 3':<34} acumulado {100 * a:+7.0f}% · caída máx. {100 * dd:5.0f}%")
    res = {}
    for nom, opsd in EST.items():
        eq = sum(curva_porcion(opsd[m], dias, ini) for m in datos) / len(datos); a, dd = met(eq); res[nom] = (a, dd)
        L.append(f"   {nom + ' · porción':<34} acumulado {100 * a:+7.0f}% · caída máx. {100 * dd:5.0f}%")
    for nom in ("Círculo verde (todos)", "Tu regla (Strong + ATR 2%)"):
        eq = sim_riesgo(EST[nom], dias, ini); a, dd = met(eq)
        L.append(f"   {nom + ' · riesgo 1%':<34} acumulado {100 * a:+7.0f}% · caída máx. {100 * dd:5.0f}%  (con US$ 4.000: {a * 4000:+,.0f})".replace(",", "."))
    L += ["", f"== POR MITADES, las tres juntas en porción (operaciones que empiezan antes y después del {CORTE.date()}) =="]
    d1 = pd.date_range(ini, CORTE, freq="D"); d2 = pd.date_range(CORTE, ult, freq="D")
    for nom, opsd in EST.items():
        e1 = sum(curva_porcion([o for o in opsd[m] if o["e"] < CORTE], d1, ini) for m in datos) / len(datos)
        e2 = sum(curva_porcion(opsd[m], d2, CORTE) for m in datos) / len(datos)
        (a1, x1), (a2, x2) = met(e1), met(e2)
        L.append(f"   {nom:<28} 1.ª mitad {100 * a1:+6.0f}% (caída {100 * x1:4.0f}%) · 2.ª mitad {100 * a2:+6.0f}% (caída {100 * x2:4.0f}%)")
    L += ["", "== OPERACIONES POR AÑO (las tres monedas) =="]
    for nom, opsd in EST.items():
        anios = pd.Series([op["e"].year for m in opsd for op in opsd[m] if op["e"] >= ini]).value_counts().sort_index()
        L.append(f"   {nom:<28} " + " · ".join(f"{a}: {n}" for a, n in anios.items()))
    txt = "\n".join(L); open("data/investigacion_cripto_circulos.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
