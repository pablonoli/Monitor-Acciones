"""Investigación (no es parte del dashboard): baja historia larga de Yahoo (desde 2015) y prueba estrategias de acciones
en distintos períodos (2018-2019, 2020, 2021, 2022, 2023 en adelante) y grupos (defensivas, emergentes, resto).
Escribe data/investigacion_resultados.json y un resumen en texto en data/investigacion_resumen.txt. Se corre a mano desde Actions."""
import json, sys, os, time, datetime as dt
import numpy as np
EPOCH = dt.date(1970, 1, 1)
COST = 0.0015
TOP99 = "NVDA AAPL MSFT AMZN GOOGL AVGO META TSLA BRK-B WMT LLY JPM V ORCL XOM JNJ MA COST PLTR BAC ABBV MU HD NFLX AMD PG GE CVX UNH MS KO CAT CSCO GS WFC IBM MRK LRCX PM RTX AMAT AXP TMO INTC MCD CRM ABT TMUS C LIN APP KLAC DIS PEP BA ISRG APH SCHW BLK AMGN ACN UBER GEV TJX QCOM TXN NEE DHR T BKNG VZ SPGI ANET LOW INTU COF GILD ADI PFE DE SYK NOW UNP HON LMT BSX PANW ETN WELL ADBE MDT NEM BX CEG PLD COP PGR PH CB".split()
DEF = set("KO PEP PG JNJ MO WMT VZ T MCD HSY PFE O SNA ABBV UNH COST TGT NKE SBUX CVX XOM LMT ADP MRK ABT TMO LIN NEE PM AMGN GILD DUK SO CL KMB GIS MDLZ".split())
EM = set("BABA BIDU JD PBR ABEV GGAL YPF PAM VIST NU MELI".split())
PERIODOS = [("2018 a hoy", "2018-01-01", None), ("2018-2019", "2018-01-01", "2020-01-01"), ("2020 (covid)", "2020-01-01", "2021-01-01"),
            ("2021", "2021-01-01", "2022-01-01"), ("2022 (bajista)", "2022-01-01", "2023-01-01"), ("2023 a hoy", "2023-01-01", None)]

def dia(s): return (dt.date.fromisoformat(s) - EPOCH).days

def cargar():
    import pandas as pd, yfinance as yf
    fav = json.load(open("data/favoritas.json"))
    simbolos = list(dict.fromkeys(["SPY"] + fav + TOP99))
    out = {}
    for i in range(0, len(simbolos), 20):
        lote = simbolos[i:i + 20]
        try:
            d = yf.download(lote, start="2015-06-01", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print("lote falló", e, file=sys.stderr); continue
        for s in lote:
            try:
                x = (d[s] if isinstance(d.columns, pd.MultiIndex) else d)[["Open", "Close"]].dropna()
                if len(x) > 300:
                    out[s] = (np.array([(pd.Timestamp(t).date() - EPOCH).days for t in x.index]), x["Open"].to_numpy(float), x["Close"].to_numpy(float))
            except Exception:
                pass
    return out

def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0)); o = np.full(len(x), np.nan); o[n - 1:] = (c[n:] - c[:-n]) / n; return o
def rsi(x, n):
    d = np.diff(x, prepend=x[0]); up = np.clip(d, 0, None); dn = np.clip(-d, 0, None)
    au = np.zeros(len(x)); ad = np.zeros(len(x)); au[n] = up[1:n + 1].mean(); ad[n] = dn[1:n + 1].mean()
    for i in range(n + 1, len(x)): au[i] = (au[i - 1] * (n - 1) + up[i]) / n; ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    r = 100 - 100 / (1 + au / np.where(ad == 0, 1e-9, ad)); r[:n] = 50; return r
def E(x): return np.nan_to_num(x.astype(float), nan=0) > 0

def backtest(days, O, C, entry, exitf, start, maxhold=None, stop=None):
    n = len(C); e = 1.0; p = 0; held = 0; ent_p = 0; ent_e = 0; trades = []; expo = 0; eq = np.ones(n)
    start_i = int(np.searchsorted(days, start))
    for i in range(1, n):
        e *= 1 + p * (O[i] / C[i - 1] - 1); want = p
        if days[i] >= start:
            s = i - 1
            if p == 0 and entry[s]: want = 1
            elif p == 1 and (exitf(s, held, ent_p) or (maxhold and held >= maxhold) or (stop and C[s] <= ent_p * (1 - stop))): want = 0
        if want != p:
            e *= 1 - COST
            if want == 1: ent_p = O[i]; ent_e = e; held = 0
            else: trades.append((e / ent_e - 1, held))
            p = want
        e *= 1 + p * (C[i] / O[i] - 1); eq[i] = e
        if p == 1 and days[i] >= start: held += 1; expo += 1
    if p == 1: trades.append((e / ent_e - 1, held))
    i0 = max(0, start_i - 1); seg = eq[i0:] / eq[i0]
    return dict(ret=eq[-1] / eq[i0] - 1, trades=trades, expo=expo / max(1, n - start_i), hold=C[-1] / C[i0] - 1, dd=float((seg / np.maximum.accumulate(seg) - 1).min()))

def estrategias():
    def cruce(f, s): return lambda C, p: (E(sma(C, f) > sma(C, s)), (lambda i, h, pp, a=sma(C, f), b=sma(C, s): a[i] <= b[i]), None, None)
    def salida_trail(x):
        def fn(C, p):
            s50, s200 = sma(C, 50), sma(C, 200)
            return E(s50 > s200) & E(C > s50), (lambda i, h, pp: s50[i] <= s200[i] or C[i] <= C[max(0, i - h):i + 1].max() * (1 - x)), None, None
        return fn
    def sma_salida(n):
        def fn(C, p):
            s50, s200, sn = sma(C, 50), sma(C, 200), sma(C, n)
            return E(s50 > s200) & E(C > sn), (lambda i, h, pp: C[i] <= sn[i]), None, None
        return fn
    return {
        "Comprar y mantener": lambda C, p: (np.ones(len(C), bool), lambda i, h, pp: False, None, None),
        "Cruce simple 50/200": cruce(50, 200),
        "Cruce simple 10/30": cruce(10, 30),
        "50/200, sale si cae 15% del máximo": salida_trail(0.15),
        "50/200, sale si cae 10% del máximo": salida_trail(0.10),
        "50/200, sale bajo el promedio de 100d": sma_salida(100),
        "Precio sobre/bajo su promedio de 200d": lambda C, p: (E(C > sma(C, 200)), lambda i, h, pp: C[i] <= sma(C, 200)[i], None, None),
        "Rango 6 meses (compra <20%, vende >60%, stop 8%)": lambda C, p: (E(p < 0.2), lambda i, h, pp: p[i] > 0.6, 40, 0.08),
        "Rebote RSI(14)<30, sale >50": lambda C, p: (E(rsi(C, 14) < 30), lambda i, h, pp: rsi(C, 14)[i] > 50, 30, None),
    }

def pos_rango(C):
    n = len(C); p = np.full(n, np.nan)
    for i in range(120, n):
        w = C[i - 120:i + 1]; p[i] = (C[i] - w.min()) / (w.max() - w.min() + 1e-9)
    return p

def correr(data, periodos=PERIODOS):
    est = estrategias()
    grupos = {"Resto (no defensivas ni emergentes)": lambda s: s not in DEF and s not in EM and s != "SPY", "Defensivas": lambda s: s in DEF, "Emergentes (China, Brasil, Argentina)": lambda s: s in EM}
    pre = {s: (d, O, C, pos_rango(C)) for s, (d, O, C) in data.items()}
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9
        for gn, filtro in grupos.items():
            L = [s for s in pre if filtro(s)]
            acum = {k: [] for k in est}
            n_ok = 0
            for s in L:
                d, O, C, p = pre[s]
                k = int(np.searchsorted(d, fin)); d, O, C, p = d[:k], O[:k], C[:k], p[:k]
                if len(C) < 250 or d[0] > ini - 250 or d[-1] < ini + 40: continue
                n_ok += 1
                for nom, f in est.items():
                    ent, ex, mh, st = f(C, p)
                    acum[nom].append(backtest(d, O, C, ent, ex, ini, mh, st))
            if n_ok < 5: continue
            for nom, R in acum.items():
                tr = [t for x in R for t in x["trades"]]
                filas.append(dict(periodo=pn, grupo=gn, estrategia=nom, acciones=n_ok, ret=float(np.mean([x["ret"] for x in R])), caida=float(np.mean([x["dd"] for x in R])),
                                  ops=len(tr) / len(R), gana=float(np.mean([t[0] > 0 for t in tr])) if tr else 0.0, peor=float(min([t[0] for t in tr])) if tr else 0.0,
                                  mercado=float(np.mean([x["expo"] for x in R])), mantener=float(np.mean([x["hold"] for x in R]))))
    return filas

def resumen(filas):
    out = []; clave = None
    for f in filas:
        if (f["periodo"], f["grupo"]) != clave:
            clave = (f["periodo"], f["grupo"]); out.append(f"\n== {f['periodo']} | {f['grupo']} ({f['acciones']} acciones)")
        out.append(f"  {f['estrategia']:52s} rinde {100*f['ret']:7.1f}%  caída prom {100*f['caida']:6.1f}%  ops {f['ops']:5.1f}  gana {100*f['gana']:3.0f}%  peor {100*f['peor']:6.0f}%  mercado {100*f['mercado']:3.0f}%")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s")
    filas = correr(data)
    json.dump(filas, open("data/investigacion_resultados.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_resumen.txt", "w", encoding="utf-8").write(txt); print(txt[:3000])
