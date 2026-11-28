"""Investigación: ¿las defensivas rinden mejor "comprando la baja" que comprando la ruptura (círculo verde)? 5 años, CEDEARs.
Defensivas = misma lista de investigacion_defensivas.py. Se compara contra el resto de las acciones y contra el círculo verde con el plan A (cierra a las 20 ruedas).
Entradas (la señal se mide al cierre; se compra en la apertura siguiente). Solo con el cierre por encima de su media de 200 ruedas (tendencia sana):
  RSI : el RSI de 14 ruedas cruza por debajo de 35 (ayer estaba en 35 o más).
  CAIDA: el cierre queda 5% o más por debajo del máximo de 20 ruedas (ayer no).
Salidas (stop siempre a 2,5 ATR por debajo de la entrada; ATR en dólares = ATR % del día de la señal x cierre; máximo 20 ruedas, se vende al cierre de la rueda 20):
  REBOTE: se vende al cierre del primer día (desde la 2.ª rueda) en que el cierre supera su media de 10 ruedas.
  OBJETIVO: se vende al tocar entrada + 2 ATR.
Convenciones iguales a investigacion_circulos_20_vol.py: costo 0,15% por lado, stop antes que objetivo en la misma vela, gaps al open; capital con 1% de riesgo por operación, exposición máx. 100%.
Regla extra: no se abre una operación nueva en la misma acción mientras la anterior sigue abierta.
Resultado: data/investigacion_defensivas_baja.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_20_vol as V
import investigacion_circulos_filtros as F
from investigacion_defensivas import DEFENSIVAS, MIN_OPS

K = F.K
ENTRADAS = [("RSI", "RSI 14 cruza por debajo de 35"), ("CAIDA", "Cae 5% o más desde el máximo de 20 ruedas")]
SALIDAS = [("REBOTE", "Sale al superar su media de 10 ruedas"), ("OBJ", "Sale al tocar +2 ATR")]
STOP_ATR, OBJ_ATR, MAX_RUEDAS = 2.5, 2.0, 20


def extra(raw, P):
    c = raw["Close"]; d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    P["rsi"] = (100 - 100 / (1 + up / dn)).values; P["sma10"] = c.rolling(10).mean().values; P["hi20"] = raw["High"].rolling(20).max().values


def camino(P, e, stop, salida, obj):
    ent = P["O"][e]; n = len(P["C"]); path = []
    for j in range(e, min(n, e + MAX_RUEDAS)):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if lo <= stop:
            path.append(min(stop, o) / ent * K); return path, "stop"
        if salida == "OBJ" and hi >= obj:
            path.append(max(obj, o) / ent * K); return path, "objetivo"
        path.append(c / ent * K)
        if salida == "REBOTE" and j > e and c > P["sma10"][j]: return path, "rebote"
    return path, "tiempo"


def senal(P, i, entrada):
    c = P["C"][i]
    if np.isnan(P["sma200"][i]) or np.isnan(P["atrp"][i]) or not c > P["sma200"][i]: return False
    if entrada == "RSI": return P["rsi"][i] < 35 <= P["rsi"][i - 1]
    cai, ant = c / P["hi20"][i] - 1, P["C"][i - 1] / P["hi20"][i - 1] - 1
    return cai <= -0.05 and ant > -0.05


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    raw = I.bajar(simbolos, ini_dl); datos = {s: I.preparar(x) for s, x in raw.items()}
    for s, x in raw.items(): extra(x, datos[s])
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    defs = [s for s in DEFENSIVAS if s in datos]
    claves = [(en, sa) for en, _ in ENTRADAS for sa, _ in SALIDAS]
    filas = {k: [] for k in claves}; verde = []
    for s, P in datos.items():
        g = "def" if s in defs else "resto"; n = len(P["f"])
        libre = {k: 0 for k in claves}
        for i in range(210, n - 1):
            if P["f"][i] < ini: continue
            e = i + 1; ent = P["O"][e]
            if P["verde"][i] and not np.isnan(P["atrp"][i]) and not np.isnan(P["sma200"][i]):
                stop = P["low20"][i]
                if 0 < stop < ent:
                    path, half, siguio = V.camino(P, e, stop, 20, None, 20, 20); ret = path[-1] - 1; riesgo = (ent - stop) / ent
                    verde.append(dict(s=s, g=g, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0,
                                      path=path, half=half, dts=list(P["f"][e:e + len(path)]), dur=len(path), siguio=siguio, atrp=float(P["atrp"][i])))
            for en, sa in claves:
                if e <= libre[(en, sa)] or not senal(P, i, en): continue
                atr_d = P["atrp"][i] * P["C"][i]; stop = ent - STOP_ATR * atr_d; obj = ent + OBJ_ATR * atr_d
                if not stop > 0: continue
                path, mot = camino(P, e, stop, sa, obj); ret = path[-1] - 1; riesgo = (ent - stop) / ent
                libre[(en, sa)] = e + len(path)
                filas[(en, sa)].append(dict(s=s, g=g, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo, fl=0,
                                            path=path, half=None, dts=list(P["f"][e:e + len(path)]), dur=len(path), mot=mot, atrp=float(P["atrp"][i])))
    DF = {k: pd.DataFrame(v) for k, v in filas.items()}; DV = pd.DataFrame(verde)
    L = [f"¿LAS DEFENSIVAS RINDEN MEJOR 'COMPRANDO LA BAJA'? Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR, {len(defs)} defensivas.",
         f"Defensivas: {', '.join(defs)}.",
         "Entradas (solo con el cierre sobre su media de 200 ruedas; se compra en la apertura siguiente): " + "; ".join(f"{k} = {t}" for k, t in ENTRADAS) + ".",
         f"Salidas (stop a {STOP_ATR} ATR bajo la entrada; máximo {MAX_RUEDAS} ruedas): " + "; ".join(f"{k} = {t}" for k, t in SALIDAS) + f" (objetivo a +{OBJ_ATR} ATR).",
         f"Costo {100 * I.COSTO:.2f}% por lado. Stop antes que objetivo en la misma vela; gaps al open. Capital con 1% de riesgo por operación, exposición máx. 100%, cada grupo con su capital. No se abre otra operación en la misma acción mientras la anterior sigue abierta.",
         f"Referencia: círculo verde con plan A (cierra a las 20 ruedas) en las mismas acciones. Se probaron {len(claves)} versiones de 'comprar la baja' (sesgo de selección: la mejor suele rendir menos hacia adelante).",
         "LÍMITES: lista de CEDEARs de hoy (sesgo de supervivencia); las defensivas son pocas y sus señales también (* = menos de 200 operaciones: poco confiable); las operaciones se superponen y no son independientes; la clasificación de defensivas es mía por sector.", ""]
    enc = f"{'Versión':<46}| {'ops':>5} | {'toma.':>5} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'dur.':>4} | {'acum.':>8} | {'c. MTM':>7} | {'efec./mes':>9} | {'mes+':>4} | {'peor mes':>8}"
    for g, nom in (("def", "DEFENSIVAS"), ("resto", "RESTO")):
        L += [f"== RESULTADOS: {nom} ==", enc]
        res = {}
        sub = DV[DV["g"] == g]; x = V.metricas(sub, dias); res["verde"] = (sub, x)
        L.append(V.linea("Círculo verde, plan A (referencia)", x, "*" if x and x["ops"] < MIN_OPS else ""))
        for en, te in ENTRADAS:
            for sa, ts in SALIDAS:
                df = DF[(en, sa)]; sub = df[df["g"] == g] if len(df) else df
                x = V.metricas(sub, dias) if len(sub) else None; res[(en, sa)] = (sub, x)
                L.append(V.linea(f"{en}: {te[:28]} | {sa}", x, "*" if x and x["ops"] < MIN_OPS else ""))
        L.append("   Cómo terminan las operaciones (stop / objetivo / rebote / tiempo):")
        for en, _ in ENTRADAS:
            for sa, _ in SALIDAS:
                sub = res[(en, sa)][0]
                if len(sub): L.append(f"   {en}+{sa}: " + ", ".join(f"{m} {100 * (sub['mot'] == m).mean():.0f}%" for m in ("stop", "objetivo", "rebote", "tiempo")) + f"; duración mediana {sub['dur'].median():.0f} ruedas")
        L.append(f"-- por mitades (señales hasta {corte.date()} | desde {corte.date()}; el capital vuelve a empezar en cada mitad) --")
        for k, nom in [("verde", "verde A")] + [((en, sa), f"{en}+{sa}") for en, sa in claves]:
            sub = res[k][0]
            if not len(sub): continue
            a, b = sub[sub["f"] < corte], sub[sub["f"] >= corte]
            xa, xb = (V.metricas(a, d1) if len(a) else None), (V.metricas(b, d2) if len(b) else None)
            fa = (f"{len(a):>4}{'*' if len(a) < MIN_OPS else ' '} ops, acum. {100 * xa['acu']:+7.1f}%, R {xa['R']:+5.2f}, c. MTM {100 * xa['mtm']:+6.1f}%") if xa else "sin operaciones"
            fb = (f"{len(b):>4}{'*' if len(b) < MIN_OPS else ' '} ops, acum. {100 * xb['acu']:+7.1f}%, R {xb['R']:+5.2f}, c. MTM {100 * xb['mtm']:+6.1f}%") if xb else "sin operaciones"
            L.append(f"   {nom:<14}| 1.ª: {fa} || 2.ª: {fb}")
        L.append("")
    tot, dd, _ = F.spy_bh(ini, ult, ini_dl)
    L.append(f"Referencia: SPY comprar y mantener en el mismo período: {100 * tot:+.1f}% (caída máxima {100 * dd:.1f}%).")
    L.append("Marca de casualidad: una diferencia de acumulado menor a 15 puntos contra la referencia puede ser casualidad.")
    txt = "\n".join(L); open("data/investigacion_defensivas_baja.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
