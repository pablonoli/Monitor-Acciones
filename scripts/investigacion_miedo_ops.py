"""Investigación: las operaciones sueltas de 'comprar el miedo' (comprado SOLO porque el VIX tocó 30 y el cruce 50/200 no decía comprar)."""
import numpy as np, pandas as pd
import investigacion_factores as F
import investigacion_miedo10 as M

if __name__ == "__main__":
    F.DEMORA = 2
    d = M.cargar(); vix = d["VIXMAX"].dropna()
    L = ["OPERACIONES DE 'COMPRAR EL MIEDO' (solo las que se hacen cuando el cruce 50/200 NO decía comprar). Compra al cierre del día siguiente a la señal.", "",
         f"{'Activo':<9}{'Operaciones':>12}{'Duración mediana':>18}{'Más larga':>11}{'Ganó':>7}{'Rend. mediano':>15}{'Mejor':>9}{'Peor':>9}"]
    for a in M.ACT:
        px = d[a].dropna(); v = vix.reindex(px.index, method="ffill")
        base = F.sma(px, 50) > F.sma(px, 200); mie = (v > 30) & ~base | False
        pos = (base | (v > 30)).shift(2).fillna(False); solo = (pos & ~base.shift(2).fillna(False)).values
        r = px.pct_change().fillna(0.0).values; ops = []; i = 0; n = len(solo)
        while i < n:
            if solo[i]:
                j = i; eq = 1.0
                while j < n and solo[j]: eq *= 1 + r[j]; j += 1
                ops.append((j - i, eq - 1)); i = j
            else: i += 1
        if not ops: continue
        du = np.array([o[0] for o in ops]); rt = np.array([o[1] for o in ops])
        L.append(f"{a:<9}{len(ops):>12}{np.median(du):>14.0f} d{du.max():>9} d{(rt > 0).mean() * 100:>6.0f}%{F.pct(np.median(rt)):>15}{F.pct(rt.max()):>9}{F.pct(rt.min()):>9}")
    txt = "\n".join(L); open("data/investigacion_miedo_ops.txt", "w", encoding="utf-8").write(txt); print(txt)
