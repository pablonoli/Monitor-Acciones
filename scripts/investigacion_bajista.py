"""Investigación (no es parte del dashboard): ¿qué estrategia funciona mejor en mercados bajistas?
Períodos de caída (2000-02, 2008, covid 2020, 2022) con ~100 acciones grandes + favoritas. Escribe data/investigacion_bajista.txt/json."""
import time, json, sys
import numpy as np
import datetime as dt
from investigacion import sma, E, dia, EPOCH, TOP99, DEF
from investigacion_niveles import pos_rango, en_rango, maq
from investigacion_parciales import bt_frac
from investigacion_nuevo_cruce import fijada_nuevo

COMP, VEND, STOP, NV = 0.2, 0.6, 0.08, 120
PERIODOS = [("2000-2002 (burbuja punto com)", "2000-03-01", "2002-10-31"), ("2008 (crisis)", "2007-10-01", "2009-03-31"),
            ("2008 + rebote hasta mar-2010", "2007-10-01", "2010-03-31"), ("2020 (covid)", "2020-02-01", "2020-04-30"),
            ("2022 (bajista)", "2022-01-01", "2022-10-31"), ("2022 + rebote hasta fin 2023", "2022-01-01", "2023-12-31")]

def cargar():
    import pandas as pd, yfinance as yf
    fav = json.load(open("data/favoritas.json"))
    simbolos = list(dict.fromkeys(["SPY"] + fav + TOP99 + list(DEF)))
    out = {}
    for i in range(0, len(simbolos), 20):
        lote = simbolos[i:i + 20]
        try: d = yf.download(lote, start="1998-01-01", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
        except Exception as e: print("lote falló", e, file=sys.stderr); continue
        for s in lote:
            try:
                x = (d[s] if isinstance(d.columns, pd.MultiIndex) else d)[["Open", "Close"]].dropna()
                if len(x) > 300: out[s] = (np.array([(pd.Timestamp(t).date() - EPOCH).days for t in x.index]), x["Open"].to_numpy(float), x["Close"].to_numpy(float))
            except Exception: pass
    return out

NOMS = ["Comprar y mantener", "Quedarse en efectivo", "Entrar de a poco (1/6 por mes) y mantener", "Cruce 50/200", "Automática (rango + cruce)",
        "Cruce 50/200 + solo si el SPY está sobre su media 200", "Automática + solo si el SPY está sobre su media 200"]

def correr(data):
    spy_d, spy_O, spy_C = data["SPY"]; spy_ok_s = E(spy_C > sma(spy_C, 200))
    filas = []; spy_filas = []
    pre = {}
    for s, (d, O, C) in data.items():
        if s == "SPY": continue
        fl = en_rango(C); cr = E(sma(C, 50) > sma(C, 200))
        ix = np.clip(np.searchsorted(spy_d, d, "right") - 1, 0, len(spy_d) - 1)
        pre[s] = dict(d=d, O=O, C=C, cr=cr, fl=fl, pos=pos_rango(C, NV), spy=spy_ok_s[ix])
    for (pn, a, b) in PERIODOS:
        ini = dia(a); fin = dia(b)
        for grupo in ("todas", "defensivas"):
            acum = {}
            for s, v in pre.items():
                if grupo == "defensivas" and s not in DEF: continue
                k = int(np.searchsorted(v["d"], fin)); d, O, C = v["d"][:k], v["O"][:k], v["C"][:k]
                if len(C) < 300 or d[0] > ini - 360 or d[-1] < ini + 40: continue
                cr, fl, pos, sp = v["cr"][:k], v["fl"][:k], v["pos"][:k], v["spy"][:k]
                r = maq(d, O, C, pos, ini, COMP, VEND, STOP) > 0
                auto = fijada_nuevo(fl, r, cr)
                rampa = np.clip((d - ini) / 180.0, 0, 1)
                res = {NOMS[0]: np.ones(k), NOMS[1]: np.zeros(k), NOMS[2]: rampa, NOMS[3]: cr.astype(float), NOMS[4]: auto.astype(float),
                       NOMS[5]: (cr & sp).astype(float), NOMS[6]: (auto & sp).astype(float)}
                for nom, t in res.items(): acum.setdefault(nom, []).append(bt_frac(d, O, C, t, ini))
            n = len(acum.get(NOMS[0], []))
            if n < (8 if grupo == "defensivas" else 15): continue
            for nom in NOMS:
                R = acum[nom]; rets = np.array([x["ret"] for x in R]); dds = np.array([x["dd"] for x in R])
                filas.append(dict(periodo=pn, grupo=grupo, acciones=n, estrategia=nom, prom=float(rets.mean()), mediana=float(np.median(rets)), dd=float(np.median(dds)),
                                  gana=float(np.mean(rets > 0)), expo=float(np.mean([x["expo"] for x in R]))))
        # El índice SPY solo
        k = int(np.searchsorted(spy_d, fin))
        if spy_d[0] <= ini - 360:
            d, O, C = spy_d[:k], spy_O[:k], spy_C[:k]; ok = spy_ok_s[:k]; cr = E(sma(C, 50) > sma(C, 200))
            for nom, t in {"SPY: comprar y mantener": np.ones(k), "SPY: efectivo cuando cae bajo su media 200": ok.astype(float),
                           "SPY: cruce 50/200": cr.astype(float), "SPY: vender en corto cuando cae bajo su media 200": np.where(ok, 1.0, -1.0),
                           "SPY: entrar de a poco (1/6 por mes)": np.clip((d - ini) / 180.0, 0, 1)}.items():
                x = bt_frac(d, O, C, t, ini); spy_filas.append(dict(periodo=pn, estrategia=nom, ret=x["ret"], dd=x["dd"], expo=x["expo"]))
    return filas, spy_filas

def texto(filas, spy):
    out = []
    for pn, _, _ in PERIODOS:
        for g in ("todas", "defensivas"):
            F = [f for f in filas if f["periodo"] == pn and f["grupo"] == g]
            if not F: continue
            out.append(f"== {pn} · {g} ({F[0]['acciones']} acciones)")
            for f in F: out.append(f"  {f['estrategia'][:62]:62} prom {100*f['prom']:6.1f}%  mediana {100*f['mediana']:6.1f}%  caída típica {100*f['dd']:6.1f}%  ganan {100*f['gana']:3.0f}%  invertido {100*f['expo']:3.0f}%")
        S = [f for f in spy if f["periodo"] == pn]
        if S:
            out.append(f"== {pn} · índice SPY")
            for f in S: out.append(f"  {f['estrategia'][:62]:62} retorno {100*f['ret']:6.1f}%  caída máx {100*f['dd']:6.1f}%  invertido {100*f['expo']:3.0f}%")
    return "\n".join(out)

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos bajados en", round(time.time() - t0), "s", flush=True)
    filas, spy = correr(data)
    json.dump(dict(filas=filas, spy=spy), open("data/investigacion_bajista.json", "w"), ensure_ascii=False)
    txt = texto(filas, spy); open("data/investigacion_bajista.txt", "w", encoding="utf-8").write(txt); print(txt)
