"""Consulta: todas las señales verdes (y Strong) de IBIT, con resultado Plan A. Solo escribe data/consulta_ibit.txt en esta rama."""
import sys
import pandas as pd
sys.path.insert(0, "scripts")
import tendencia as T  # noqa: E402

df = T.bajar_uno("IBIT")
x = df.dropna(subset=["Open", "High", "Low", "Close"])
c, h, lo, v = x["Close"], x["High"], x["Low"], x["Volume"]
techo = h.shift(1).rolling(20).max()
mom = c.diff()
z = (mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)
sma = c.rolling(200).mean()
rupt = (c > techo) & (c.shift(1) <= techo.shift(1)) & (z > 1.5)
verde = rupt & (c > sma)
strong = verde & (z > 2.5) & (v > 1.5 * v.rolling(20).mean())
tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
atrp = (tr / c).rolling(14).mean()
stop20 = lo.rolling(20).min()
f = lambda k: x.index[k].date().strftime("%d/%m/%Y")

def plan_a(i):
    if i + 1 >= len(x):
        return "compra mañana"
    e = float(x["Open"].iloc[i + 1]); st = float(stop20.iloc[i]); R = e - st
    mitad, res, txt = False, 0.0, f"compra {f(i+1)} a {e:.2f}, stop {st:.2f}"
    for k in range(i + 1, min(i + 21, len(x))):
        o, hh, ll, cc = (float(x[col].iloc[k]) for col in ("Open", "High", "Low", "Close"))
        if ll <= st:
            s = min(o, st); res += (0.5 if mitad else 1) * (s - e) / R
            return txt + f"; vende {f(k)} a {s:.2f} ({'stop' if not mitad else 'al costo'}) -> {res:+.2f}R = {res*40000:+,.0f} pesos ({res*R/e*100:+.1f}%)".replace(",", ".")
        if not mitad and hh >= e + 2 * R:
            res += 1; mitad = True; st = e; txt += f"; mitad {f(k)} a {e+2*R:.2f}"
        if k == i + 20:
            res += (0.5 if mitad else 1) * (cc - e) / R
            return txt + f"; vende {f(k)} a {cc:.2f} (rueda 20) -> {res:+.2f}R = {res*40000:+,.0f} pesos ({res*R/e*100:+.1f}%)".replace(",", ".")
    return txt + f"; sigue abierta, hoy {float(c.iloc[-1]):.2f}"

out = [f"IBIT: datos desde {f(0)} hasta {f(len(x)-1)}. Hoy {float(c.iloc[-1]):.2f}, SMA 200 {float(sma.iloc[-1]):.2f}", "",
       "Señales verdes (ruptura con fuerza y arriba de la SMA 200):"]
for i in [k for k in range(len(x)) if bool(verde.iloc[k])]:
    out.append(f"{f(i)}  {'STRONG' if strong.iloc[i] else 'verde '}  Z {z.iloc[i]:.2f}  ATR {atrp.iloc[i]*100:.1f}%  cierre {c.iloc[i]:.2f}  |  {plan_a(i)}")
out += ["", "Rupturas con fuerza que el filtro bloqueó por estar debajo de la SMA 200 (últimas 10):"]
for i in [k for k in range(len(x)) if bool(rupt.iloc[k]) and not bool(verde.iloc[k])][-10:]:
    out.append(f"{f(i)}  Z {z.iloc[i]:.2f}  cierre {c.iloc[i]:.2f}  SMA200 {sma.iloc[i]:.2f}")
open("data/consulta_ibit.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
