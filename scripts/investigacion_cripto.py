"""Investigación (no es parte del dashboard): ¿qué medias conviene para las cripto? Compara cruces de medias exponenciales
(10/30, 20/50, 20/100, 50/100, 50/200) con y sin el filtro "BTC o propio" (sobre el promedio de 200 días), contra comprar y mantener.
Escribe data/investigacion_cripto.txt. Se corre sola cuando cambia este archivo o a mano desde Actions."""
import sys, time, json
import numpy as np
from investigacion import backtest, sma, dia, EPOCH

SIMB = ["BTC-USD", "ETH-USD", "SOL-USD", "HYPE-USD", "MSTR", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "COIN"]
PARES = [(10, 30), (20, 50), (20, 100), (50, 100), (50, 200)]
PERIODOS = [("2018 a hoy", "2018-01-01", None), ("2021 a hoy", "2021-01-01", None), ("2022 (bajista)", "2022-01-01", "2023-01-01"),
            ("2023 a hoy", "2023-01-01", None), ("2024 a hoy (el del panel)", "2024-01-01", None)]

def cargar():
    import pandas as pd, yfinance as yf
    out = {}
    for s in SIMB:
        try:
            x = yf.download(s, start="2014-09-01", interval="1d", auto_adjust=False, progress=False)
            if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
            x = x[["Open", "Close"]].dropna()
            if len(x) > 300:
                out[s] = (np.array([(pd.Timestamp(t).date() - EPOCH).days for t in x.index]), x["Open"].to_numpy(float), x["Close"].to_numpy(float))
        except Exception as e:
            print(s, "falló", e, file=sys.stderr)
    return out

def ema(x, n):
    a = 2 / (n + 1); o = np.empty(len(x)); o[0] = x[0]
    for i in range(1, len(x)): o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o

def correr(data, periodos=PERIODOS):
    bd, bO, bC = data["BTC-USD"]; bflag = bC > np.nan_to_num(sma(bC, 200), nan=1e18)
    pre = {}
    for s, (d, O, C) in data.items():
        j = np.clip(np.searchsorted(bd, d, side="right") - 1, 0, len(bd) - 1)
        permiso = bflag[j] | (C > np.nan_to_num(sma(C, 200), nan=1e18))
        pre[s] = dict(d=d, O=O, C=C, permiso=permiso, cruce={p: ema(C, p[0]) > ema(C, p[1]) for p in PARES}, lenta={p: p[1] for p in PARES})
    filas = []
    for (pn, a, b) in periodos:
        ini = dia(a); fin = dia(b) if b else 10 ** 9; acum = {}
        for s, v in pre.items():
            k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
            if len(C) < 300 or d[0] > ini - 250 or d[-1] < ini + 40: continue
            def bt(sen): return backtest(d, O, C, sen, lambda i, h, pp, sen=sen: not sen[i], ini)
            base = bt(np.zeros(len(C), bool)); hold = base["hold"]
            i0 = int(np.searchsorted(d, ini)); seg = C[i0:] / C[i0]
            res = {"Comprar y mantener": dict(ret=hold, hold=hold, expo=1.0, trades=[], dd=float((seg / np.maximum.accumulate(seg) - 1).min()))}
            for p in PARES:
                c = v["cruce"][p][:k]
                for nomf, sen in (("sin filtro", c), ("con filtro BTC o propio", c & v["permiso"][:k])):
                    res[f"EMA {p[0]}/{p[1]} {nomf}"] = bt(sen)
            for nom, x in res.items(): acum.setdefault(nom, []).append(x)
        n_ok = len(acum.get("Comprar y mantener", []))
        if n_ok < 4: continue
        for nom, R in acum.items():
            azar = float(np.mean([(1 + x["hold"]) ** x["expo"] - 1 for x in R])); ret = float(np.mean([x["ret"] for x in R]))
            filas.append(dict(periodo=pn, activos=n_ok, estrategia=nom, ret=ret, vs_azar=ret - azar, ops=float(np.mean([len(x["trades"]) for x in R])),
                              caida=float(np.mean([x["dd"] for x in R])), mercado=float(np.mean([x["expo"] for x in R]))))
    return filas

def resumen(filas):
    out = []; clave = None
    for f in filas:
        if f["periodo"] != clave: clave = f["periodo"]; out.append(f"\n== {f['periodo']} ({f['activos']} activos)")
        out.append(f"  {f['estrategia']:36s} rinde {100*f['ret']:8.1f}%  vs azar {100*f['vs_azar']:+7.1f}  caída prom {100*f['caida']:6.1f}%  ops {f['ops']:5.1f}  en el mercado {100*f['mercado']:3.0f}%")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(sorted(data), "bajados en", round(time.time() - t0), "s", flush=True)
    filas = correr(data); json.dump(filas, open("data/investigacion_cripto.json", "w"), ensure_ascii=False)
    txt = resumen(filas); open("data/investigacion_cripto.txt", "w", encoding="utf-8").write(txt); print(txt)
