"""Investigación: ¿el MSB+ del indicador de TradingView ("MSB-OB · SR", estilo LuxAlgo) le gana al círculo verde como entrada? 5 años, CEDEARs.
Interpretación del indicador de TV (configuración de Pablo: ZigZag 9, Fib Factor 0,33, cajas de 80 velas, Strong = cuerpo >= 0,7 ATR):
- Zigzag: la tendencia pasa a bajista cuando el mínimo de la vela es el más bajo de 9 velas, y a alcista cuando el máximo es el más alto de 9 velas.
  Al cambiar, se fija el pivote del tramo anterior (techo o piso).
- MSB+ (en tiempo real): el cierre supera el último techo h1 + 0,33 x (h1 - piso posterior l0). Una sola señal por techo.
- Caja verde (Bu-OB): la última vela bajista (apertura > cierre) entre el techo h1 y el piso l0. Strong si su cuerpo >= 0,7 ATR(14).
- Caja roja (Be-OB) para el filtro: al MSB−, la última vela alcista entre el piso l1 y el techo h0; activa 80 velas o hasta que un cierre la atraviese.
Salida igual para todas (Plan A): mitad a +2R y stop al precio de compra, cierre de todo en la rueda 20; stop antes que objetivo en la misma vela; gaps al open.
Capital: 1% de riesgo por operación, exposición máx. 100%, y ninguna posición mayor al 25% del capital (si el stop está muy cerca, se arriesga menos del 1%).
Resultado: data/investigacion_msb_tv.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F
import investigacion_circulos_20_sel as S

K = F.K
ZZ, FIB, OBLEN, ESPERA, MAXTAM = 9, 0.33, 80, 20, 0.25
CHICA = 0.15


def camino(P, e, ent, stop, intradia=False, tope=20):
    """Valor por unidad invertida (neto de costos) al cierre de cada rueda. intradia: la compra se hace dentro de la rueda e (orden límite):
    ese día solo cuenta el stop (supuesto pesimista), no el objetivo."""
    R = ent - stop; obj = ent + 2 * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []; hk = None
    for j in range(e, min(n, e + tope)):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if intradia and j == e: o = ent
        if lo <= sc:
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path, ((hk, vm) if mitad else None)
        if not mitad and hi >= obj and not (intradia and j == e):
            vm = max(obj, o) / ent * K; mitad = True; sc = ent; hk = j - e
        path.append(0.5 * vm + 0.5 * (c / ent * K) if mitad else c / ent * K)
    return path, ((hk, vm) if mitad else None)


def estructura(x):
    """Devuelve listas de eventos MSB+ y MSB− (en tiempo real, al cierre) con sus cajas."""
    o, h, l, c = (x[k].values for k in ("Open", "High", "Low", "Close")); n = len(c)
    hh = pd.Series(h).rolling(ZZ).max().values; ll = pd.Series(l).rolling(ZZ).min().values
    cp = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - cp), np.abs(l - cp)]), axis=0); atr = pd.Series(tr).rolling(14).mean().values
    trend = 1; ext_v, ext_i = h[0], 0; piv = []   # piv: (tipo 'H'/'L', idx, valor)
    usado_h, usado_l = None, None; market = 0; alc, baj = [], []
    for i in range(ZZ, n):
        if trend == 1 and l[i] <= ll[i]:
            piv.append(("H", ext_i, ext_v)); trend = -1; ext_v, ext_i = l[i], i
        elif trend == -1 and h[i] >= hh[i]:
            piv.append(("L", ext_i, ext_v)); trend = 1; ext_v, ext_i = h[i], i
        else:
            if trend == 1 and h[i] >= ext_v: ext_v, ext_i = h[i], i
            if trend == -1 and l[i] <= ext_v: ext_v, ext_i = l[i], i
        # MSB+: último techo confirmado h1 y piso confirmado posterior l0
        hs = [p for p in piv if p[0] == "H"]; ls = [p for p in piv if p[0] == "L"]
        if hs and ls:
            h1 = hs[-1]; l0 = ls[-1]
            if l0[1] > h1[1] and usado_h != h1[1] and c[i] > h1[2] + FIB * abs(h1[2] - l0[2]):
                usado_h = h1[1]
                k = next((j for j in range(l0[1], h1[1] - 1, -1) if o[j] > c[j]), l0[1])
                alc.append(dict(i=i, cambio=(market == -1), top=h[k], bot=l[k], strong=bool(abs(c[k] - o[k]) >= 0.7 * atr[k]) if not np.isnan(atr[k]) else False))
                market = 1
            l1 = ls[-1]; h0 = hs[-1]
            if h0[1] > l1[1] and usado_l != l1[1] and c[i] < l1[2] - FIB * abs(h0[2] - l1[2]):
                usado_l = l1[1]
                k = next((j for j in range(h0[1], l1[1] - 1, -1) if c[j] > o[j]), h0[1])
                baj.append(dict(i=i, top=h[k], bot=l[k]))
                market = -1
    return alc, baj


def fila(s, P, i, e, ent, stop, intradia, atrp, strong, z):
    path, half = camino(P, e, ent, stop, intradia)
    riesgo = (ent - stop) / ent; ret = path[-1] - 1
    return dict(s=s, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=max(riesgo, 0.01 / MAXTAM), rreal=riesgo, ret=ret, rnet=ret / riesgo, fl=0,
                path=path, half=half, dts=list(P["f"][e:e + len(path)]), dur=len(path), atrp=atrp, strong=strong, z=z)


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    crudos = I.bajar(simbolos, ini_dl); datos = {s: I.preparar(x) for s, x in crudos.items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); corte = pd.Timestamp("2024-05-26")
    dias = [x for x in sx.index if ini <= x <= ult]; d1 = [x for x in dias if x < corte]; d2 = [x for x in dias if x >= corte]
    sb_acu, sb_dd, sb_por = F.spy_bh(ini, ult, ini_dl)
    V = {k: [] for k in ("1", "2", "3", "4", "5", "6", "7", "8", "9")}
    ret_stats = dict(msb=0, volvio=0, rota=0, nunca=0)
    for s, P in datos.items():
        x = crudos[s]; c = x["Close"]; mom = c.diff(); Z = ((mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)).values
        n = len(P["f"]); alc, baj = estructura(x)
        sobre = lambda i: (not np.isnan(P["sma200"][i])) and P["C"][i] > P["sma200"][i]
        msb_dias = set(a["i"] for a in alc)
        # cajas rojas activas por día (para el filtro 9)
        rojas = []
        for b in baj:
            fin = min(n, b["i"] + OBLEN)
            for j in range(b["i"] + 1, fin):
                if P["C"][j] > b["top"]: fin = j; break
            rojas.append((b["i"], fin, b["bot"]))
        for i in range(210, n - 1):
            if P["f"][i] < ini or np.isnan(P["atrp"][i]): continue
            atrp, z = float(P["atrp"][i]), float(Z[i]); e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if P["verde"][i] and 0 < stop < ent:
                f1 = fila(s, P, i, e, ent, stop, False, atrp, bool(P["strong"][i]), z); V["1"].append(f1)
                if any(j in msb_dias for j in range(i - 5, i + 1)): V["8"].append(f1)
                R = P["C"][i] - stop
                if not any(a <= i < b and P["C"][i] < bot <= P["C"][i] + 2 * R for a, b, bot in rojas): V["9"].append(f1)
        for a in alc:
            i = a["i"]
            if i < 210 or i >= n - 1 or P["f"][i] < ini or np.isnan(P["atrp"][i]): continue
            atrp, z = float(P["atrp"][i]), float(Z[i]); e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if 0 < stop < ent:
                f2 = fila(s, P, i, e, ent, stop, False, atrp, False, z); V["2"].append(f2)
                if sobre(i): V["3"].append(f2)
                if sobre(i) and a["cambio"]: V["4"].append(f2)
            # retroceso a la caja: orden límite en el techo de la caja durante 20 ruedas; se cancela si un cierre (o la apertura) queda debajo de la caja
            ret_stats["msb"] += 1
            top, bot = a["top"], a["bot"]; stop2 = bot * 0.999; hecho = None
            if not (stop2 < top < P["C"][i]): ret_stats["nunca"] += 1; continue
            for j in range(i + 1, min(n, i + 1 + ESPERA)):
                if P["O"][j] < bot: hecho = "rota"; break
                if P["L"][j] <= top: hecho = j; break
                if P["C"][j] < bot: hecho = "rota"; break
            if hecho is None: ret_stats["nunca"] += 1; continue
            if hecho == "rota": ret_stats["rota"] += 1; continue
            j = hecho; ent2 = min(P["O"][j], top)
            if not (stop2 < ent2): continue
            ret_stats["volvio"] += 1
            f5 = fila(s, P, i, j, ent2, stop2, True, atrp, a["strong"], z); V["5"].append(f5)
            if sobre(i): V["6"].append(f5)
            if sobre(i) and a["strong"]: V["7"].append(f5)
    NOM = {"1": "1. Círculo verde (tu sistema, base)",
           "2": "2. MSB+ de TV, sin filtro de tendencia",
           "3": "3. MSB+ de TV + arriba de la SMA 200",
           "4": "4. MSB+ de TV solo cambio de tendencia + SMA 200",
           "5": "5. Retroceso a la caja verde, sin filtro",
           "6": "6. Retroceso a la caja verde + SMA 200",
           "7": "7. Retroceso a caja Strong + SMA 200",
           "8": "8. Círculo verde + MSB+ en las últimas 5 ruedas",
           "9": "9. Círculo verde sin caja roja dentro de 2R arriba"}
    DF = {k: pd.DataFrame(v) for k, v in V.items()}
    X = {k: (S.met(df, dias) if len(df) else None) for k, df in DF.items()}
    meses = len(pd.period_range(dias[0], dias[-1], freq="M"))
    L = [f"¿EL MSB+ DE TRADINGVIEW LE GANA AL CÍRCULO VERDE? Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR.",
         "Salida igual para todas (Plan A): mitad a +2R y stop al precio de compra, cierre de todo en la rueda 20. Costo 0,15% por lado.",
         "Entradas 1-4, 8, 9: apertura del día siguiente, stop en el mínimo de 20 ruedas. Entradas 5-7: orden límite en el techo de la caja durante 20 ruedas, stop debajo de la caja.",
         f"Capital: 1% de riesgo por operación, exposición máx. 100%, posición máx. {int(100 * MAXTAM)}% del capital. Si no hay plata libre, la señal se saltea.",
         f"Indicador de TV interpretado con tu configuración: ZigZag {ZZ}, Fib Factor {FIB}, Strong = cuerpo de la vela >= 0,7 ATR. Es una reconstrucción: el código real puede diferir.",
         f"Retroceso a la caja: de {ret_stats['msb']} MSB+, el precio volvió a la caja en {ret_stats['volvio']} ({100 * ret_stats['volvio'] / max(1, ret_stats['msb']):.0f}%); "
         f"en {ret_stats['nunca']} no volvió en 20 ruedas y en {ret_stats['rota']} la caja se rompió sin dar entrada.",
         f"Marca '≈base' = diferencia de acumulado contra la 1 menor a {int(100 * CHICA)} puntos: puede ser casualidad. 9 versiones probadas.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%.",
         "LÍMITES: sesgo de selección (varias versiones), sesgo de supervivencia (lista de CEDEARs de hoy), operaciones superpuestas, reconstrucción del indicador de TV.", ""]
    enc = (f"{'Versión':<52}| {'señales':>7} | {'toma.':>5} | {'por mes':>7} | {'prom.':>7} | {'gan.':>4} | {'R med':>5} | {'acum.':>8} | {'c. MTM':>7} | {'peor mes':>8} | vs base")
    L += ["== RESULTADOS (5 años) ==", enc]
    base = X["1"]["acu"]
    for k in NOM:
        x = X[k]
        if x is None: L.append(f"{NOM[k]:<52}| sin operaciones"); continue
        dif = x["acu"] - base; nota = "" if k == "1" else f"{100 * dif:+6.1f} pp{'  ≈base' if abs(dif) < CHICA else ''}"
        L.append(f"{NOM[k]:<52}| {x['sen']:>7} | {x['tom']:>5} | {x['pmes']:>7.1f} | {100 * x['prom']:+6.2f}% | {100 * x['gan']:3.0f}% | {x['R']:+5.2f} | "
                 f"{100 * x['acu']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {100 * x['peor']:+7.1f}% | {nota}")
    L.append("señales = todas las que aparecieron; toma. = las que entraron con el capital disponible; R med = resultado promedio medido en riesgos.")
    anios_l = sorted({a for x in X.values() if x for a in x["res"]["por"]} | set(sb_por))
    L += ["", "== RETORNO DEL CAPITAL POR AÑO ==", f"{'Versión':<52}" + "".join(f"{a:>10}" for a in anios_l)]
    for k in NOM:
        x = X[k]
        if x: L.append(f"{NOM[k]:<52}" + "".join(f"{100 * x['res']['por'][a]:+9.1f}%" if a in x["res"]["por"] else f"{'—':>10}" for a in anios_l))
    L.append(f"{'SPY comprar y mantener':<52}" + "".join(f"{100 * sb_por[a]:+9.1f}%" if a in sb_por else f"{'—':>10}" for a in anios_l))
    L += ["", f"== POR MITADES (señales hasta {corte.date()} | desde {corte.date()}) ==",
          f"{'Versión':<52}| {'1.ª acum.':>9} | {'por mes':>7} | {'R med':>5} | {'c. MTM':>7} || {'2.ª acum.':>9} | {'por mes':>7} | {'R med':>5} | {'c. MTM':>7}"]
    M1, M2 = {}, {}
    for k in NOM:
        df = DF[k]
        if not len(df): continue
        xa, xb = S.met(df[df["f"] < corte], d1), S.met(df[df["f"] >= corte], d2)
        if not xa or not xb: L.append(f"{NOM[k]:<52}| sin operaciones en alguna mitad"); continue
        M1[k], M2[k] = xa["acu"], xb["acu"]
        L.append(f"{NOM[k]:<52}| {100 * xa['acu']:+8.1f}% | {xa['pmes']:>7.1f} | {xa['R']:+5.2f} | {100 * xa['mtm']:+6.1f}% || {100 * xb['acu']:+8.1f}% | {xb['pmes']:>7.1f} | {xb['R']:+5.2f} | {100 * xb['mtm']:+6.1f}%")
    ambas = [k for k in M1 if k != "1" and k in M2 and M1[k] > M1.get("1", 9) and M2[k] > M2.get("1", 9)]
    L.append(f"  Le ganan al círculo verde (1) en las DOS mitades: {', '.join(ambas) or 'ninguna'}.")
    txt = "\n".join(L); open("data/investigacion_msb_tv.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
