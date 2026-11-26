"""Investigación: ¿qué conviene hacer a las 20 ruedas con los círculos verdes? 5 años, CEDEARs.
Mismas convenciones que investigacion_circulos_planes.py / investigacion_circulos_filtros.py: entrada en la apertura siguiente a la señal, costo 0,15% por lado,
stop inicial = mínimo de 20 ruedas, el stop se revisa antes que el objetivo en la misma vela, gaps al open (si abre debajo del stop se vende a la apertura),
vende la mitad a +2R y mueve el stop al precio de compra; capital con 1% de riesgo por operación, exposición máxima 100%.

Versiones (todas con el mismo plan base; la rueda 1 es la de entrada):
 A. Cerrar todo al cierre de la rueda 20, siempre.
 B. Al cierre de la rueda 20: si el precio está por debajo (o igual) del de compra, cerrar; si está arriba, seguir con el stop en el precio de compra hasta la rueda 40.
 C. Igual que B, pero seguir hasta la rueda 60.
 D. Al cierre de la rueda 20: cerrar si está por debajo de +1R; si está en +1R o más, seguir con el stop en el precio de compra hasta la rueda 40.
 E. Referencia: sin decisión en la rueda 20, máximo 60 ruedas (el plan original).
Cada una sin filtro y con SPY por encima de su SMA 200 el día de la señal. Resultado: data/investigacion_circulos_20.txt"""
import json, sys
import numpy as np, pandas as pd
import investigacion_circulos as I
import investigacion_circulos_filtros as F

K = F.K
DEC = 20   # rueda de la decisión
CORTO = {"A": "cierra en la 20", "B": "si gana sigue hasta 40", "C": "si gana sigue hasta 60", "D": "si está en +1R sigue hasta 40", "E": "sin límite (60)"}
# (letra, nombre, tope si no sigue, regla en la rueda 20, tope si sigue)
VERS = [("A", "Cerrar todo a las 20 ruedas, siempre", 20, None, 20),
        ("B", "A las 20: si gana, sigue con stop en compra hasta 40", 20, "pos", 40),
        ("C", "A las 20: si gana, sigue con stop en compra hasta 60", 20, "pos", 60),
        ("D", "A las 20: si está en +1R o más, sigue hasta 40", 20, "1R", 40),
        ("E", "Referencia: sin límite en la 20 (máx. 60)", 60, None, 60)]


def camino(P, e, stop, tope, regla, tope_sigue):
    """Valor de la operación (por unidad invertida, neto de costos) al cierre de cada rueda; el último es el realizado.
    Devuelve (path, half, siguio): half = (rueda en que vendió la mitad, valor de esa mitad) o None; siguio = True si pasó la rueda 20 por la regla."""
    ent = P["O"][e]; R = ent - stop; obj = ent + 2 * R; sc = stop; mitad = False; vm = 0.0; n = len(P["C"]); path = []; hk = None
    fin = e + tope; siguio = False
    j = e
    while j < min(n, fin):
        o, hi, lo, c = P["O"][j], P["H"][j], P["L"][j], P["C"][j]
        if lo <= sc:                                            # stop primero (con gap: vende a la apertura si abrió debajo)
            v = min(sc, o) / ent * K
            path.append(0.5 * vm + 0.5 * v if mitad else v); return path, ((hk, vm) if mitad else None), siguio
        if not mitad and hi >= obj:
            vm = max(obj, o) / ent * K; mitad = True; sc = ent; hk = j - e
        v = c / ent * K
        path.append(0.5 * vm + 0.5 * v if mitad else v)
        if regla and j - e + 1 == DEC:                          # cierre de la rueda 20: decidir
            sigue = c > ent if regla == "pos" else c >= ent + R
            if sigue:
                siguio = True; fin = e + tope_sigue; sc = max(sc, ent)
            else:
                return path, ((hk, vm) if mitad else None), siguio
        j += 1
    return path, ((hk, vm) if mitad else None), siguio


def metricas(df, dias):
    res = F.sim(df); c = F.caja_datos(res, dias); mm = F.mercado(res, dias)
    r, R = df["ret"].values, df["rnet"].values
    return dict(res=res, c=c, mm=mm, ops=len(df), tom=res["tom"], prom=r.mean(), med=np.median(r), gan=(r > 0).mean(), R=R.mean(),
                dur=np.median(df["dur"].values), acu=res["acu"], mtm=mm[0], efec=c["inflow"].mean() if c else np.nan,
                mesp=(c["pnl"] > 0).mean() if c else np.nan, peor=c["pnl"].min() if c else np.nan, ganmes=c["pnl"].mean() if c else np.nan)


def main(anios=5):
    d = json.load(open("data/tendencia_acciones.json")); simbolos = sorted(d["cedear"].keys())
    ini_dl = (pd.Timestamp.today() - pd.DateOffset(years=anios + 2)).strftime("%Y-%m-%d")
    datos = {s: I.preparar(x) for s, x in I.bajar(simbolos, ini_dl).items()}
    sx = I.bajar(["SPY"], ini_dl)["SPY"]; spy_ok_s = (sx["Close"] > sx["Close"].rolling(200).mean())
    ult = max(P["f"][-1] for P in datos.values()); ini = ult - pd.DateOffset(years=anios); mitad_t = ini + (ult - ini) / 2
    dias = [x for x in sx.index if ini <= x <= ult]
    sb_acu, sb_dd, sb_por = F.spy_bh(ini, ult, ini_dl)
    filas = {v[0]: [] for v in VERS}
    for s, P in datos.items():
        ok = spy_ok_s.reindex(P["f"], method="ffill").fillna(False).values; n = len(P["f"])
        for i in range(210, n - 1):
            if P["f"][i] < ini or not P["verde"][i] or np.isnan(P["atrp"][i]) or np.isnan(P["sma200"][i]): continue
            e = i + 1; ent = P["O"][e]; stop = P["low20"][i]
            if not (0 < stop < ent): continue
            riesgo = (ent - stop) / ent
            for letra, _, tope, regla, tope_sigue in VERS:
                path, half, siguio = camino(P, e, stop, tope, regla, tope_sigue); ret = path[-1] - 1
                v20 = path[DEC - 1] if len(path) >= DEC else None           # valor al cierre de la rueda 20 (si llegó)
                abierta = (e + len(path) - 1 == n - 1) and not (len(path) >= (tope_sigue if siguio else tope))
                filas[letra].append(dict(s=s, f=P["f"][i], fe=P["f"][e], fx=P["f"][e + len(path) - 1], riesgo=riesgo, ret=ret, rnet=ret / riesgo,
                                         fl=(1 if ok[i] else 0), path=path, half=half, dts=list(P["f"][e:e + len(path)]), dur=len(path),
                                         siguio=siguio, v20=v20, abierta=abierta))
    L = [f"¿QUÉ HACER A LAS 20 RUEDAS CON LOS CÍRCULOS VERDES? Últimos {anios} años ({ini.date()} a {ult.date()}). {len(datos)} acciones con CEDEAR.",
         f"Plan base: entrada en la apertura siguiente, stop inicial = mínimo de 20 ruedas, mitad a +2R y stop al precio de compra. Costo {100 * I.COSTO:.2f}% por lado. Stop antes que objetivo en la misma vela; gaps al open.",
         "Capital: 1% de riesgo por operación (tamaño = 1% / distancia al stop), sin apalancamiento (exposición máx. 100%: si no hay lugar, se saltea la señal), se actualiza al cerrar.",
         "Columnas: ops = señales; toma. = operaciones realmente tomadas con el capital; prom./med./gan./R = por operación (cada una cuenta igual); dur. = mediana de ruedas hasta cerrar;",
         "acum. = resultado acumulado del capital; c. MTM = peor caída valuando las abiertas a mercado; efec./mes = capital que vuelve a efectivo por mes; mes+ = % de meses con ganancia realizada; peor mes = peor mes realizado.",
         f"SPY comprar y mantener: acumulado {100 * sb_acu:+.1f}%, caída máx. {100 * sb_dd:+.1f}%.",
         "LÍMITES: se probaron solo 5 variantes elegidas de antemano y se muestra cuál dio mejor (sesgo de selección: la ganadora probablemente rinda algo menos de lo que muestra);",
         "la lista de CEDEARs es la de hoy (sesgo de supervivencia: faltan acciones que se hundieron o dejaron de cotizar); las operaciones se superponen y no son independientes.", ""]
    enc = f"{'Versión':<56}| {'ops':>4} | {'toma.':>5} | {'prom.':>7} | {'med.':>7} | {'gan.':>4} | {'R med':>5} | {'dur.':>4} | {'acum.':>8} | {'c. MTM':>7} | {'efec./mes':>9} | {'gan./mes':>8} | {'mes+':>4} | {'peor mes':>8}"
    tot = {}
    for titulo, m in (("SIN FILTRO", 0), ("SPY POR ENCIMA DE SU SMA 200", 1)):
        L += [f"== {titulo} ==", enc]
        for letra, nombre, *_ in VERS:
            df = pd.DataFrame(filas[letra]); df = df[(df["fl"] & m) == m]
            x = metricas(df, dias); tot[(letra, m)] = (df, x)
            L.append(f"{letra}. {nombre:<53}| {x['ops']:>4} | {x['tom']:>5} | {100 * x['prom']:+6.2f}% | {100 * x['med']:+6.2f}% | {100 * x['gan']:3.0f}% | {x['R']:+5.2f} | {x['dur']:>4.0f} | {100 * x['acu']:+7.1f}% | {100 * x['mtm']:+6.1f}% | {100 * x['efec']:8.0f}% | {100 * x['ganmes']:+7.2f}% | {100 * x['mesp']:3.0f}% | {100 * x['peor']:+7.1f}%")
        L.append("")
    # por año
    anios_l = sorted({a for (_, _), (_, x) in tot.items() for a in x["res"]["por"]} | set(sb_por))
    L += ["== RETORNO DEL CAPITAL POR AÑO (1% de riesgo por operación; por año de cierre de las operaciones; SPY: retorno total del año calendario) ==",
          f"{'Versión':<62}" + "".join(f"{a:>10}" for a in anios_l)]
    for titulo, m in (("sin filtro", 0), ("SPY>SMA200", 1)):
        for letra, nombre, *_ in VERS:
            por = tot[(letra, m)][1]["res"]["por"]
            L.append(f"{(letra + '. ' + CORTO[letra] + ' (' + titulo + ')'):<62}" + "".join(f"{100 * por[a]:+9.1f}%" if a in por else f"{'—':>10}" for a in anios_l))
    L.append(f"{'SPY comprar y mantener':<62}" + "".join(f"{100 * sb_por[a]:+9.1f}%" if a in sb_por else f"{'—':>10}" for a in anios_l))
    L.append("")
    # dos mitades
    L += [f"== ROBUSTEZ: LOS 5 AÑOS PARTIDOS EN DOS MITADES (primera: señales hasta {mitad_t.date()}; segunda: desde ahí). El capital vuelve a empezar (100%) en cada mitad. ==",
          f"{'Versión':<56}| {'1.ª: acum.':>10} | {'R med':>5} | {'c. MTM':>7} | {'mes+':>4} || {'2.ª: acum.':>10} | {'R med':>5} | {'c. MTM':>7} | {'mes+':>4}"]
    d1 = [x for x in dias if x < mitad_t]; d2 = [x for x in dias if x >= mitad_t]
    rank = {}
    for titulo, m in (("sin filtro", 0), ("SPY>SMA200", 1)):
        for letra, nombre, *_ in VERS:
            df = tot[(letra, m)][0]; a, b = df[df["f"] < mitad_t], df[df["f"] >= mitad_t]
            xa, xb = metricas(a, d1), metricas(b, d2)
            rank.setdefault(m, []).append((letra, xa["acu"], xb["acu"], xa["R"], xb["R"]))
            L.append(f"{(letra + '. ' + CORTO[letra] + ' (' + titulo + ')'):<56}| {100 * xa['acu']:+9.1f}% | {xa['R']:+5.2f} | {100 * xa['mtm']:+6.1f}% | {100 * xa['mesp']:3.0f}% || {100 * xb['acu']:+9.1f}% | {xb['R']:+5.2f} | {100 * xb['mtm']:+6.1f}% | {100 * xb['mesp']:3.0f}%")
    for m, t in ((0, "sin filtro"), (1, "SPY>SMA200")):
        rk = rank[m]
        o1 = " > ".join(v[0] for v in sorted(rk, key=lambda v: -v[1])); o2 = " > ".join(v[0] for v in sorted(rk, key=lambda v: -v[2]))
        L.append(f"  Orden por acumulado ({t}): primera mitad {o1} | segunda mitad {o2}")
    L.append("")
    # después de la rueda 20
    L += ["== QUÉ PASÓ DESPUÉS DE LA RUEDA 20 (operaciones que seguían abiertas al cierre de la rueda 20) ==",
          "Ganancia después de la 20 = cambio del valor de la operación entre el cierre de la rueda 20 y la salida final, en % del precio de compra y en R."]
    for titulo, m in (("sin filtro", 0), ("SPY>SMA200", 1)):
        for letra, nombre, _, regla, _ in VERS:
            df = tot[(letra, m)][0]
            if letra == "E":
                g = df[df["dur"] > DEC]; tx = "seguían abiertas después de la 20 (sin regla)"
            elif regla:
                g = df[df["siguio"]]; tx = "siguieron por la regla"
            else:
                continue
            if not len(g):
                L.append(f"  {letra} ({titulo}): ninguna"); continue
            post = np.array([(p[-1] - p[DEC - 1]) for p in g["path"]]); postR = post / g["riesgo"].values
            llega = df[df["dur"] >= DEC]
            L.append(f"  {letra} ({titulo}): {len(g)} de {len(df)} operaciones {tx} ({100 * len(g) / max(1, len(llega)):.0f}% de las que llegaron vivas a la 20). "
                     f"Después de la 20: promedio {100 * post.mean():+.2f}% ({postR.mean():+.2f} R), mediana {100 * np.median(post):+.2f}%, ganaron más {100 * (post > 0).mean():.0f}%, "
                     f"mejor {100 * post.max():+.1f}%, peor {100 * post.min():+.1f}%; ruedas extra mediana {np.median(g['dur'].values - DEC):.0f}.")
    # las que se cerraron en la 20 estando en pérdida o ganancia (para A)
    for titulo, m in (("sin filtro", 0), ("SPY>SMA200", 1)):
        df = tot[("A", m)][0]; g = df[df["dur"] == DEC]
        if len(g):
            L.append(f"  A ({titulo}): {len(g)} operaciones se cerraron por tiempo en la rueda 20; de ellas {100 * (g['ret'] > 0).mean():.0f}% con ganancia, promedio {100 * g['ret'].mean():+.2f}%.")
    L.append("")
    L.append("Nota: 'gana' en B/C = cierre de la rueda 20 por encima del precio de compra; en D = cierre de la rueda 20 en precio de compra + 1R o más. Al seguir, el stop pasa al precio de compra (si ya estaba ahí por la venta de la mitad, queda igual).")
    txt = "\n".join(L); open("data/investigacion_circulos_20.txt", "w", encoding="utf-8").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
