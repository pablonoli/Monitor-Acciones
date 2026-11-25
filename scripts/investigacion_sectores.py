"""Investigación (no es parte del dashboard): rotación entre los 11 sectores del S&P 500 (ETFs que tienen CEDEAR).
¿Conviene ir hacia el sector que "repunta" frente al mercado? Compara varias reglas contra comprar SPY y mantener."""
import sys, numpy as np, pandas as pd

SEC = ["XLK", "XLV", "XLE", "XLF", "XLU", "XLP", "XLY", "XLI", "XLB", "XLRE", "XLC"]
NOM = {"XLK": "Tecnología", "XLV": "Salud", "XLE": "Energía", "XLF": "Financiero", "XLU": "Servicios públicos", "XLP": "Consumo básico",
       "XLY": "Consumo discrecional", "XLI": "Industriales", "XLB": "Materiales", "XLRE": "Inmobiliario", "XLC": "Comunicaciones"}
COSTO = 0.0015          # 0,15% por lado, sobre la parte que se compra o vende
INICIO = "2008-01-01"


def cargar():
    import yfinance as yf
    d = yf.download(SEC + ["SPY"], start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    return d.dropna(how="all").ffill(limit=3)


def sma(x, n): return x.rolling(n, min_periods=n).mean()


def estado(entrada, mantiene):
    """Máquina de estados por columna: entra cuando 'entrada' es True y se queda mientras 'mantiene' sea True."""
    e = entrada.values; m = mantiene.values; a = np.zeros(e.shape, dtype=bool)
    for t in range(1, len(e)):
        a[t] = (a[t - 1] & m[t]) | (e[t] & m[t])
    return pd.DataFrame(a, index=entrada.index, columns=entrada.columns)


def pesos_de(activo, cols, tope=4):
    """Partes iguales entre los sectores activos (cada uno máximo 1/tope); lo que sobra va a SPY."""
    n = activo.sum(axis=1)
    w = activo.astype(float).div(np.maximum(n, tope), axis=0)
    w["SPY"] = 1 - w.sum(axis=1)
    return w


def reglas(px):
    s = px[SEC]; spy = px["SPY"]
    rs = s.div(spy, axis=0)
    rs50 = sma(rs, 50)
    sobre = rs > rs50
    # "giro": el sector pasó de ir peor a ir mejor que el mercado en los últimos 10 días
    giro = sobre & ((~sobre).astype(float).rolling(10).max().fillna(0) > 0)
    px50 = s > sma(s, 50)
    r6 = rs / rs.shift(126) - 1
    ok = s.notna() & rs50.notna()
    out = {}
    # E1: giro frente al mercado; se sale cuando vuelve a quedar peor que su promedio
    out["Repunte: giro vs mercado (E1)"] = pesos_de(estado(giro & ok, sobre & ok), SEC)
    # E2: igual, y el sector además sube por sí solo (precio sobre su promedio de 50)
    out["Repunte + sube solo (E2)"] = pesos_de(estado(giro & px50 & ok, sobre & px50 & ok), SEC)
    # E3: igual que E2, pero solo entra si todavía no corrió (ventaja contra el mercado en 6 meses menor a 10%)
    out["Repunte temprano: no corrió aún (E3)"] = pesos_de(estado(giro & px50 & (r6 < 0.10) & ok, sobre & px50 & ok), SEC)
    # M1 / M2: mensual, los 3 mejores por rendimiento (lo clásico: va a lo que ya subió)
    fin = s.index.to_series().groupby([s.index.year, s.index.month]).max()
    def top3(score):
        w = pd.DataFrame(0.0, index=s.index, columns=SEC + ["SPY"]);
        for f in fin.values:
            sc = score.loc[f].dropna()
            if len(sc) < 3: w.loc[f, "SPY"] = 1.0; continue
            for c in sc.nlargest(3).index: w.loc[f, c] = 1 / 3
        mask = pd.Series(s.index.isin(fin.values), index=s.index)
        w[~mask] = np.nan
        return w.ffill().fillna(0.0).assign(SPY=lambda d: 1 - d[SEC].sum(axis=1))
    out["Mensual: 3 mejores de 6 meses (M1)"] = top3(s / s.shift(126) - 1)
    mix = (rs / rs.shift(21) - 1 + rs / rs.shift(63) - 1 + rs / rs.shift(126) - 1) / 3
    out["Mensual: 3 con mejor fuerza vs mercado (M2)"] = top3(mix.where(s > sma(s, 200)))
    # referencias
    eq = pd.DataFrame(0.0, index=s.index, columns=SEC + ["SPY"])
    for c in SEC: eq[c] = s[c].notna() / s.notna().sum(axis=1)
    out["Los 11 sectores parejos"] = eq
    spy_w = pd.DataFrame(0.0, index=s.index, columns=SEC + ["SPY"]); spy_w["SPY"] = 1.0
    out["Comprar y mantener SPY"] = spy_w
    return out, dict(rs=rs, rs50=rs50, sobre=sobre, giro=giro, r6=r6, px50=px50, mix=mix)


def retornos(w, px):
    r = px[SEC + ["SPY"]].pct_change().fillna(0.0)
    w = w.reindex(columns=SEC + ["SPY"]).fillna(0.0)
    bruto = (w.shift(1).fillna(0.0) * r).sum(axis=1)
    rot = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    return bruto - COSTO * rot, rot


def medidas(r, rot):
    r = r[r.index >= INICIO]; rot = rot[rot.index >= INICIO]
    if len(r) < 50: return None
    eq = (1 + r).cumprod(); a = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / a) - 1; dd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(252); sh = (r.mean() * 252) / vol if vol else 0
    return dict(cagr=cagr, dd=dd, sh=sh, rot=rot.sum() / 2 / a)


def tramo(r, desde, hasta):
    x = r[(r.index >= desde) & (r.index < hasta)]
    return (1 + x).prod() - 1


def pct(x): return f"{x * 100:+.1f}%".replace(".", ",")


def informe(px):
    reg, aux = reglas(px)
    filas = []; ret = {}
    for n, w in reg.items():
        r, rot = retornos(w, px); ret[n] = r; m = medidas(r, rot); filas.append((n, m, w))
    L = [f"ROTACIÓN ENTRE SECTORES (11 ETFs con CEDEAR). Datos hasta {px.index[-1].date()}. Costo {COSTO * 100:.2f}% por lado.",
         "Cuando ninguna regla marca sector, el dinero queda en SPY. Se opera al día siguiente de la señal.", ""]
    L.append(f"{'Regla':<46}{'Anual':>8}{'Peor caída':>12}{'Rot./año':>10}")
    for n, m, _ in filas:
        L.append(f"{n:<46}{pct(m['cagr']):>8}{pct(m['dd']):>12}{m['rot']:>10.1f}")
    L += ["", "POR PERÍODO (rendimiento total de cada tramo)"]
    tr = [("2008-01-01", "2013-01-01", "2008-12"), ("2013-01-01", "2020-01-01", "2013-19"), ("2020-01-01", "2023-01-01", "2020-22"), ("2023-01-01", "2100-01-01", "2023-hoy")]
    L.append(f"{'Regla':<46}" + "".join(f"{t[2]:>10}" for t in tr))
    for n in reg:
        L.append(f"{n:<46}" + "".join(f"{pct(tramo(ret[n], a, b)):>10}" for a, b, _ in tr))
    # estado de hoy
    L += ["", "ESTADO HOY (por sector)", f"{'Sector':<22}{'1 mes':>8}{'3 meses':>9}{'6 meses':>9}  {'vs mercado 6m':>14}  {'Giro':>5}  {'Estado'}"]
    s = px[SEC]; spy = px["SPY"]; ult = px.index[-1]
    act = {n: reg[n].iloc[-1] for n in reg}
    filasH = []
    for c in SEC:
        if s[c].dropna().empty: continue
        f = lambda k: s[c].iloc[-1] / s[c].iloc[-1 - k] - 1
        rel6 = aux["r6"][c].iloc[-1]
        e3 = act["Repunte temprano: no corrió aún (E3)"].get(c, 0) > 0
        e1 = act["Repunte: giro vs mercado (E1)"].get(c, 0) > 0
        est = "REPUNTA (entra E3)" if e3 else ("repunta pero ya corrió" if e1 else ("más fuerte que el mercado" if aux["sobre"][c].iloc[-1] else "más débil que el mercado"))
        filasH.append((aux["mix"][c].iloc[-1], f"{NOM[c] + ' ' + c:<22}{pct(f(21)):>8}{pct(f(63)):>9}{pct(f(126)):>9}  {pct(rel6):>14}  {'sí' if aux['giro'][c].iloc[-1] else 'no':>5}  {est}"))
    L += [x[1] for x in sorted(filasH, key=lambda z: -z[0])]
    return "\n".join(L)


if __name__ == "__main__":
    px = cargar()
    print(px.tail(3)); txt = informe(px)
    open("data/investigacion_sectores.txt", "w", encoding="utf-8").write(txt); print(txt)
