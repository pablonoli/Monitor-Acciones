"""Investigación: stop por volatilidad (entrada x (1 - k x ATR %), como el simulador) contra stop amplio (mínimo de 20 ruedas). 5 años, CEDEARs, Plan A.
Entrada en la apertura siguiente; mitad a +2R y stop al precio de compra; cierre de todo en la rueda 20; stop antes que objetivo en la misma vela; gaps al open.
Capital: 1% de riesgo por operación, exposición máx. 100%, posición máx. 25% del capital (con stops cortos la posición es grande).
Se prueba con todos los círculos verdes y con la opción 7 (Strong + ATR >= 2%). Resultado: data/investigacion_stop_atr.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F
import investigacion_circulos_20_sel as S
import investigacion_msb_tv as M

CHICA = 0.15


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    crudos = I.bajar(simbolos, ini_dl); datos = {s: I.preparar(x) for s, x in crudos.items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    sb_acu, sb_dd, sb_por = F.spy_bh(ini, ult, ini_dl)
    STOPS = [("min", "Amplio: mínimo de 20 ruedas"), (1.5, "Volatilidad 1,5 ATR"), (2.0, "Volatilidad 2,0 ATR"), (2.5, "Volatilidad 2,5 ATR (el del simulador)")]
    V = {k: [] for k, _ in STOPS}; dist = {k: [] for k, _ in STOPS}
    for s, P in datos.items():
        c = crudos[s]["Close"]; mom = c.diff(); Z = ((mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)).values
        n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]): continue
            e = i + 1; ent = P["O"][e]; atrp = float(P["atrp"][i])
            for k, _ in STOPS:
                stop = P["low20"][i] if k == "min" else ent * (1 - k * atrp)
                if not (0 < stop < ent): continue
                V[k].append(M.fila(s, P, i, e, ent, stop, False, atrp, bool(P["strong"][i]), float(Z[i]))); dist[k].append((ent - stop) / ent)
    L = [f"STOP POR VOLATILIDAD CONTRA STOP AMPLIO. Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR. Plan A.",
         "Stop por volatilidad = entrada x (1 - k x ATR %), con ATR % del día de la señal (igual que el simulador). Amplio = mínimo de las últimas 20 ruedas.",
         f"Capital: 1% de riesgo por operación, exposición máx. 100%, posición máx. {int(100 * M.MAXTAM)}% del capital. Costo 0,15% por lado.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%.",
         f"Marca '≈base' = diferencia de acumulado contra el stop amplio menor a {int(100 * CHICA)} puntos: puede ser casualidad. 8 versiones probadas.",
         "LÍMITES: sesgo de selección, sesgo de supervivencia (lista de hoy), operaciones superpuestas.", ""]
    for grupo, filtro in (("TODOS LOS CÍRCULOS VERDES", lambda df: df), ("OPCIÓN 7: STRONG + ATR >= 2%", lambda df: df[df["strong"] & (df["atrp"] >= 0.02)])):
        L += [f"== {grupo} ==", f"{'Stop':<42}| {'dist. media':>11} | {'toma.':>5} | {'por mes':>7} | {'prom.':>7} | {'gan.':>4} | {'stop%':>5} | {'R med':>5} | {'acum.':>8} | {'c. MTM':>7} | {'1.ª mitad':>9} | {'2.ª mitad':>9} | vs amplio"]
        base = None
        for k, nom in STOPS:
            df = pd.DataFrame(V[k]); df["dist"] = dist[k]; df = filtro(df)
            x = S.met(df, dias); a = S.met(df[df["f"] < corte], d1); b = S.met(df[df["f"] >= corte], d2)
            if x is None: L.append(f"{nom:<42}| sin operaciones"); continue
            a = a or dict(acu=float("nan")); b = b or dict(acu=float("nan"))
            t = x["res"]["filas"]; tocados = (t["rnet"] <= -0.9).mean()
            if base is None: base = x["acu"]; nota = ""
            else: dif = x["acu"] - base; nota = f"{100 * dif:+6.1f} pp{'  ≈base' if abs(dif) < CHICA else ''}"
            L.append(f"{nom:<42}| {100 * df['dist'].mean():10.1f}% | {x['tom']:>5} | {x['pmes']:>7.1f} | {100 * x['prom']:+6.2f}% | {100 * x['gan']:3.0f}% | {100 * tocados:4.0f}% | {x['R']:+5.2f} | "
                     f"{100 * x['acu']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {100 * a['acu']:+8.1f}% | {100 * b['acu']:+8.1f}% | {nota}")
        L.append("dist. media = distancia promedio de la entrada al stop; stop% = operaciones que terminaron tocando el stop inicial (−1R); R med = resultado promedio en riesgos.")
        L.append("")
    txt = "\n".join(L); open("data/investigacion_stop_atr.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
