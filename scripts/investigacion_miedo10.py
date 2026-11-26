"""Investigación: estrategia de cruce 50/200 con y sin 'comprar el miedo' (también comprado si VIX > 30) en 10 activos."""
import time, pandas as pd
import investigacion_factores as F

ACT = ["AAPL", "NVDA", "TSLA", "AMZN", "JPM", "KO", "XOM", "MELI", "SPY", "BTC-USD"]

def cargar():
    import yfinance as yf
    t = ACT + ["^VIX"]
    for i in range(5):
        d = yf.download(t, start="2000-01-01", auto_adjust=True, progress=False, threads=False)["Close"]
        if all(c in d.columns and d[c].notna().sum() > 500 for c in t): return d
        print("reintento", i + 1, flush=True); time.sleep(10)
    raise SystemExit("faltan datos")

if __name__ == "__main__":
    d = cargar(); vix = d["^VIX"].dropna(); L = [f"CRUCE 50/200 CON Y SIN 'COMPRAR EL MIEDO' (VIX > 30). Datos hasta {d.index[-1].date()}.", "",
        f"{'Activo':<9}{'Desde':<8}{'Anual solo':>11}{'Anual+miedo':>12}{'Caída solo':>12}{'Caída+miedo':>12}{'Mantener':>10}{'Comp. solo':>11}{'Comp.+miedo':>12}"]
    for a in ACT:
        px = d[a].dropna(); v = vix.reindex(px.index, method="ffill")
        base = F.sma(px, 50) > F.sma(px, 200); mie = base | (v > 30)
        res = {}
        for n, s in (("solo", base), ("miedo", mie), ("mant", pd.Series(True, index=px.index))):
            r, pos, rot = F.simular(px, s); res[n] = F.medidas(r, pos, rot)
        L.append(f"{a:<9}{str(px.index[0].year):<8}{F.pct(res['solo']['cagr']):>11}{F.pct(res['miedo']['cagr']):>12}{F.pct(res['solo']['dd']):>12}{F.pct(res['miedo']['dd']):>12}{F.pct(res['mant']['cagr']):>10}{res['solo']['tiempo'] * 100:>10.0f}%{res['miedo']['tiempo'] * 100:>11.0f}%")
    txt = "\n".join(L); open("data/investigacion_miedo10.txt", "w", encoding="utf-8").write(txt); print(txt)
