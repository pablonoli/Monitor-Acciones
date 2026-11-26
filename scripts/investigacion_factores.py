"""Investigación (no es parte del dashboard): ¿sirven el VIX, las tasas, el dólar y la época del año para operar semanas/meses?
Acciones (SPY, QQQ) y criptos (BTC, ETH, SOL) + MSTR. Dos partes: (1) qué pasó DESPUÉS según cada condición, (2) si usarlo como filtro mejora la estrategia de cruce 50/200."""
import numpy as np, pandas as pd

COSTO = 0.0015
ACT = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "SOL-USD", "MSTR"]
HALVING = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]


def cargar():
    import yfinance as yf
    t = ACT + ["^VIX", "^TNX", "^IRX", "DX-Y.NYB"]
    d = yf.download(t, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    return d.dropna(how="all")


def pct(x): return "  n/d" if pd.isna(x) else f"{x * 100:+.1f}%".replace(".", ",")


def fwd(px, n): return px.shift(-n) / px - 1


def tabla_cond(px, a, cond, nombres, hs=(21, 63)):
    """Rendimiento medio y % de veces positivo a h días, según la condición del día (cond: Series con etiquetas)."""
    L = [f"{'':<34}{'días':>7}" + "".join(f"{'a ' + str(h) + 'd':>9}{'%pos':>6}" for h in hs)]
    base = [fwd(px[a], h).dropna() for h in hs]
    c = cond.reindex(px.index)
    for nom in nombres:
        idx = c[c == nom].index; row = f"{nom:<34}{len(idx):>7}"
        for f in base:
            x = f.reindex(idx).dropna()
            row += f"{pct(x.mean()):>9}" + (f"{(x > 0).mean() * 100:>5.0f}%" if len(x) else "   n/d")
        L.append(row)
    row = f"{'(todos los días)':<34}{px[a].dropna().shape[0]:>7}"
    for f in base: row += f"{pct(f.mean()):>9}{(f > 0).mean() * 100:>5.0f}%"
    L.append(row); return L


def parte1(d):
    L = []; vix = d["^VIX"]
    # 1) VIX
    cat = pd.cut(vix, [0, 15, 20, 25, 30, 40, 200], labels=["VIX < 15", "VIX 15-20", "VIX 20-25", "VIX 25-30", "VIX 30-40", "VIX > 40"]).astype(str)
    sp = (vix > vix.rolling(50).mean() * 1.3).map({True: "pico de miedo (VIX 30% sobre su promedio)", False: "sin pico"})
    for a in ["SPY", "BTC-USD"]:
        L += ["", f"== {a}: rendimiento DESPUÉS, según el VIX del día =="]
        px = d[[a]].dropna(); L += tabla_cond(px, a, cat, list(cat.unique()[:0]) or ["VIX < 15", "VIX 15-20", "VIX 20-25", "VIX 25-30", "VIX 30-40", "VIX > 40"])
        L += tabla_cond(px, a, sp, ["pico de miedo (VIX 30% sobre su promedio)"])[1:2]
    # 2) tasas y dólar
    tn, dx = d["^TNX"], d["DX-Y.NYB"]; curva = d["^TNX"] - d["^IRX"]
    conds = {
        "Tasa 10 años sube (sobre su prom. 200d)": (tn > tn.rolling(200).mean()).map({True: "sube", False: "baja"}),
        "Tasa 10 años: cambio de 3 meses": (tn - tn.shift(63)).apply(lambda x: np.nan if pd.isna(x) else ("sube más de 0,5 pt" if x > 0.5 else ("baja más de 0,5 pt" if x < -0.5 else "estable"))),
        "Dólar (DXY) sobre su promedio de 200d": (dx > dx.rolling(200).mean()).map({True: "dólar fuerte", False: "dólar débil"}),
        "Dólar (DXY): cambio de 3 meses": (dx / dx.shift(63) - 1).apply(lambda x: np.nan if pd.isna(x) else ("sube más de 3%" if x > .03 else ("baja más de 3%" if x < -.03 else "estable"))),
        "Curva de tasas (10 años menos 3 meses)": curva.apply(lambda x: np.nan if pd.isna(x) else ("invertida" if x < 0 else "normal")),
    }
    for a in ["SPY", "BTC-USD"]:
        L += ["", f"== {a}: rendimiento DESPUÉS, según tasas y dólar =="]
        px = d[[a]].dropna()
        for tit, c in conds.items():
            L.append(f"-- {tit}"); L += tabla_cond(px, a, c.astype(object), [x for x in c.dropna().unique()])[1:-1]
    # 3) épocas
    L += ["", "== ÉPOCA DEL AÑO: rendimiento medio por mes y % de años en que ganó =="]
    for a in ["SPY", "BTC-USD"]:
        m = d[a].dropna().resample("ME").last().pct_change().dropna(); g = m.groupby(m.index.month)
        L.append(f"{a}: " + " | ".join(f"{['E', 'F', 'M', 'A', 'My', 'J', 'Jl', 'Ag', 'S', 'O', 'N', 'D'][k - 1]} {pct(v.mean())} ({(v > 0).mean() * 100:.0f}%)" for k, v in g))
        inv = m[(m.index.month >= 11) | (m.index.month <= 4)]; ver = m[(m.index.month >= 5) & (m.index.month <= 10)]
        L.append(f"   nov-abril: {pct(inv.mean())}/mes ({len(inv)} meses) · mayo-octubre: {pct(ver.mean())}/mes ({len(ver)} meses)")
    b = d["BTC-USD"].dropna(); L.append("BTC por tiempo desde el halving (rendimiento medio a 63 días, pocas observaciones: 3 ciclos):")
    f63 = fwd(b, 63).dropna()
    for i, h in enumerate(HALVING[:-1] + HALVING[-1:]):
        pass
    desde = pd.Series(np.nan, index=b.index)
    for h in HALVING: desde[b.index >= pd.Timestamp(h)] = ((b.index[b.index >= pd.Timestamp(h)] - pd.Timestamp(h)).days / 30.4)
    for lo, hi in ((0, 12), (12, 24), (24, 36), (36, 48)):
        x = f63[(desde.reindex(f63.index) >= lo) & (desde.reindex(f63.index) < hi)]
        L.append(f"   {lo}-{hi} meses después del halving: {pct(x.mean())} a 63 días ({len(x)} días, {(x > 0).mean() * 100:.0f}% positivos)")
    return L


def sma(x, n): return x.rolling(n, min_periods=n).mean()


def simular(px, senal):
    r = px.pct_change().fillna(0.0); pos = senal.shift(1).fillna(False).astype(float)
    rot = pos.diff().abs().fillna(0.0)
    return pos * r - COSTO * rot, pos, rot


def medidas(r, pos, rot, desde=None, hasta=None):
    x = r.copy(); p = pos
    if desde: x = x[x.index >= desde]; p = p[p.index >= desde]
    if hasta: x = x[x.index < hasta]; p = p[p.index < hasta]
    if len(x) < 100: return None
    a = len(x) / 365.25 if x.index.dayofweek.max() == 6 else len(x) / 252
    eq = (1 + x).cumprod(); dd = (eq / eq.cummax() - 1).min(); vol = x.std() * np.sqrt(len(x) / a)
    return dict(cagr=eq.iloc[-1] ** (1 / a) - 1, dd=dd, sh=(x.mean() * len(x) / a) / vol if vol else 0, tiempo=p.mean(), op=rot.reindex(x.index).sum() / 2 / a)


def parte2(d):
    L = ["", "== COMO FILTRO de la estrategia de cruce 50/200 (se está comprado si el cruce lo indica Y se cumple el filtro) ==",
         "Anual / peor caída / veces por año que opera / % del tiempo comprado. Entre paréntesis: mitad vieja y mitad nueva (anual).", ""]
    vix, tn, dx = d["^VIX"], d["^TNX"], d["DX-Y.NYB"]
    for a in ["SPY", "QQQ", "BTC-USD", "ETH-USD", "MSTR"]:
        px = d[a].dropna(); idx = px.index
        v = vix.reindex(idx).ffill(); t = tn.reindex(idx).ffill(); x = dx.reindex(idx).ffill()
        base = sma(px, 50) > sma(px, 200)
        F = {
            "Cruce 50/200 solo": base,
            "+ VIX bajo 25": base & (v < 25),
            "+ VIX bajo 30": base & (v < 30),
            "+ VIX no sube (bajo su prom. 20d)": base & (v < sma(v, 20)),
            "+ dólar débil (bajo prom. 100d)": base & (x < sma(x, 100)),
            "+ tasa 10a no sube fuerte (3m < +0,5pt)": base & ((t - t.shift(63)) < 0.5),
            "+ VIX bajo 25 y dólar débil": base & (v < 25) & (x < sma(x, 100)),
            "+ solo nov-abril": base & ((idx.month >= 11) | (idx.month <= 4)),
            "Comprar y mantener": pd.Series(True, index=idx),
        }
        mid = idx[len(idx) // 2]
        L.append(f"### {a} ({idx[0].date()} a {idx[-1].date()})")
        L.append(f"{'Regla':<42}{'Anual':>8}{'Caída':>8}{'Op/año':>8}{'Comp.':>7}   {'vieja':>7}{'nueva':>7}")
        for n, s in F.items():
            r, pos, rot = simular(px, s)
            m = medidas(r, pos, rot); m1 = medidas(r, pos, rot, hasta=mid); m2 = medidas(r, pos, rot, desde=mid)
            if m: L.append(f"{n:<42}{pct(m['cagr']):>8}{pct(m['dd']):>8}{m['op']:>8.1f}{m['tiempo'] * 100:>6.0f}%   {pct(m1['cagr']) if m1 else '':>7}{pct(m2['cagr']) if m2 else '':>7}")
        L.append("")
    return L


if __name__ == "__main__":
    d = cargar(); print(d.tail(2))
    txt = "\n".join([f"FACTORES QUE AFECTAN EL PRECIO. Datos hasta {d.index[-1].date()}."] + parte1(d) + parte2(d))
    open("data/investigacion_factores.txt", "w", encoding="utf-8").write(txt); print(txt)
