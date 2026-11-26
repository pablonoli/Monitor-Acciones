"""Investigación: 'comprar el miedo' a plazo largo. Se entra cuando el VIX TOCA más de 30 (y el cruce 50/200 no dice comprar) y se mantiene un tiempo fijo
(o hasta que el cruce confirme). Compra al cierre del día siguiente a la señal."""
import numpy as np, pandas as pd
import investigacion_factores as F
import investigacion_miedo10 as M


def con_tenencia(base, toca, H, hasta_cruce=False, maximo=126):
    """True si estás comprado: el cruce dice comprar, o hay una 'tenencia por miedo' activa.
    Cada toque de 30 reinicia el contador de H ruedas. Con hasta_cruce, la tenencia dura hasta que el cruce pase a comprar (tope 'maximo' ruedas)."""
    b = base.values; t = toca.values; n = len(b); out = np.zeros(n, dtype=bool); resto = 0
    for i in range(n):
        if t[i] and not b[i] and not hasta_cruce: resto = H
        if hasta_cruce:
            if t[i] and not b[i]: resto = maximo
            if b[i]: resto = 0
            out[i] = b[i] or resto > 0
            if resto > 0: resto -= 1
        else:
            out[i] = b[i] or resto > 0
            if resto > 0: resto -= 1
    return pd.Series(out, index=base.index)


if __name__ == "__main__":
    F.DEMORA = 2
    d = M.cargar(); vix = d["VIXMAX"].dropna()
    casos = [("Solo cruce", None), ("Miedo: 2 días (lo anterior)", "corto"), ("Miedo: 1 mes (21 ruedas)", 21), ("Miedo: 3 meses (63 ruedas)", 63), ("Miedo: hasta que el cruce confirme (máx. 6 meses)", "cruce")]
    L = ["COMPRAR EL MIEDO A PLAZO LARGO. Rendimiento anual / peor caída / % del tiempo comprado. Entre paréntesis: mitad vieja y mitad nueva (anual).", ""]
    for a in M.ACT:
        px = d[a].dropna(); v = vix.reindex(px.index, method="ffill"); base = F.sma(px, 50) > F.sma(px, 200); toca = v > 30; mid = px.index[len(px) // 2]
        L += [f"### {a} (desde {px.index[0].year})", f"{'Regla':<52}{'Anual':>8}{'Caída':>8}{'Comp.':>7}  {'vieja':>7}{'nueva':>7}"]
        for nom, h in casos:
            s = base if h is None else (base | toca) if h == "corto" else con_tenencia(base, toca, 0, True) if h == "cruce" else con_tenencia(base, toca, h)
            r, pos, rot = F.simular(px, s); m = F.medidas(r, pos, rot); m1 = F.medidas(r, pos, rot, hasta=mid); m2 = F.medidas(r, pos, rot, desde=mid)
            L.append(f"{nom:<52}{F.pct(m['cagr']):>8}{F.pct(m['dd']):>8}{m['tiempo'] * 100:>6.0f}%  {F.pct(m1['cagr']):>7}{F.pct(m2['cagr']):>7}")
        L.append("")
    txt = "\n".join(L); open("data/investigacion_miedo_largo.txt", "w", encoding="utf-8").write(txt); print(txt)
