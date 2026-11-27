"""Baja de Yahoo Finance las velas diarias (apertura y cierre, ajustadas por splits) de SPY y de las acciones de data/tendencia_tickers.json
y las guarda en data/tendencia_acciones.json para el dashboard de tendencia (tendencia/index.html).

Además guarda "atr": {símbolo: [ATR de 14 ruedas en USD, mínimo de los últimos 20 mínimos diarios, ATR % (promedio de rango verdadero / cierre del mismo día, 14 ruedas)]} para la columna ATR % y el simulador de posición.

Además guarda "ccl" (dólar contado con liquidación, dolarapi.com) y "cedear" ({símbolo: [último cierre en pesos del CEDEAR en BYMA vía Yahoo (ticker .BA), fecha]}) para el simulador.

Además guarda "tec": {símbolo: máximo de las últimas 20 ruedas (con la de hoy), para poner alertas de precio} y "sen": {símbolo: [fecha, strong (0/1), cierre, techo roto, zona máx., zona mín., Z-score, ruedas atrás, fecha de cierre]} con el último «círculo verde»
(ruptura alcista válida) de las últimas 5 ruedas. Reglas, con velas diarias:
 - techo = máximo de las 20 ruedas anteriores (sin la de hoy); momentum = cierre − cierre de ayer; Z = (momentum − promedio de 50) / desvío de 50 ruedas (población);
 - ruptura = cierre > techo, cierre de ayer ≤ techo de ayer y Z > 1,5; verde = ruptura con el cierre sobre la SMA 200;
 - strong = verde con Z > 2,5 y volumen > 1,5 × promedio de volumen de 20 ruedas (con la de hoy); zona = máximo y mínimo de la vela anterior a la ruptura.
 - fecha de cierre = salida por tiempo del plan: se entra en la apertura de la rueda siguiente a la señal y se cierra todo al cierre de la rueda 20 contando la de entrada
   (o sea, 20 ruedas después de la señal), salteando fines de semana y feriados de la bolsa de EE. UU.

Formato compacto: por cada símbolo, un texto con una vela por tramo separado por coma: "día.apertura.cierre", todo en base 36.
 - día: el primero es el día absoluto (días desde 1970); los demás, la diferencia con la vela anterior.
 - apertura: diferencia, en centavos, con el cierre anterior (la primera, el valor absoluto), en zigzag (negativos impares).
 - cierre: diferencia, en centavos, con la apertura de esa vela, en zigzag.
Si un símbolo falla, se conserva lo que ya estaba guardado de ese símbolo. Si falla SPY (el filtro de mercado) el script termina con error y no se guarda nada."""
import json, os, sys, time, datetime as dt
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, GoodFriday, USLaborDay, USMartinLutherKingJr, USMemorialDay,
                                    USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay

INICIO = "2023-01-01"      # hacen falta ~200 días antes de enero de 2024 para la media del permiso
RUTA_TICKERS = "data/tendencia_tickers.json"
RUTA_SALIDA = "data/tendencia_acciones.json"
LOTE = 20
EPOCH = dt.date(1970, 1, 1)
RUEDAS_TOPE = 20           # salida por tiempo: cerrar todo al cierre de la rueda 20 desde la entrada


class FeriadosNYSE(AbstractHolidayCalendar):
    """Feriados de la bolsa de Nueva York (aproximación suficiente para estimar la fecha de cierre)."""
    rules = [Holiday("Año nuevo", month=1, day=1, observance=sunday_to_monday), USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("Independencia", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday("Navidad", month=12, day=25, observance=nearest_workday)]


RUEDA = CustomBusinessDay(calendar=FeriadosNYSE())


def fecha_cierre(fecha_senal):
    """Fecha (ISO) del cierre de la rueda 20 contando desde la entrada (apertura siguiente a la señal) = 20 ruedas hábiles después de la señal."""
    return (pd.Timestamp(fecha_senal) + RUEDAS_TOPE * RUEDA).date().isoformat()


def zz(n):
    return -2 * n - 1 if n < 0 else 2 * n


def b36(n):
    n = int(n)
    if n == 0:
        return "0"
    s = ""
    while n:
        n, r = divmod(n, 36)
        s = "0123456789abcdefghijklmnopqrstuvwxyz"[r] + s
    return s


def codificar(df):
    """df: índice de fechas y columnas Open y Close. Devuelve (texto, cantidad de velas, último día, último cierre en centavos)."""
    df = df[["Open", "Close"]].dropna()
    toks, prev_dia, prev_c = [], 0, 0
    for i, (fecha, fila) in enumerate(df.iterrows()):
        dia = (pd.Timestamp(fecha).date() - EPOCH).days
        o, c = int(round(float(fila["Open"]) * 100)), int(round(float(fila["Close"]) * 100))
        toks.append(b36(dia if i == 0 else dia - prev_dia) + "." + b36(zz(o - prev_c)) + "." + b36(zz(c - o)))
        prev_dia, prev_c = dia, c
    return ",".join(toks), len(toks), prev_dia, prev_c


def volatilidad(df):
    """Devuelve [ATR de 14 ruedas en USD (promedio de Wilder), mínimo de los últimos 20 mínimos diarios, ATR %] o None.
    ATR %: para cada una de las últimas 14 ruedas, rango verdadero / cierre de ESA misma rueda, y se promedian esos porcentajes (como fracción, 0.03 = 3%)."""
    x = df[["High", "Low", "Close"]].dropna()
    if len(x) < 40:
        return None
    cierre_ant = x["Close"].shift(1)
    tr = pd.concat([x["High"] - x["Low"], (x["High"] - cierre_ant).abs(), (x["Low"] - cierre_ant).abs()], axis=1).max(axis=1).iloc[1:]
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    atrp = (tr / x["Close"].iloc[1:]).iloc[-14:].mean()
    return [round(float(atr), 3), round(float(x["Low"].iloc[-20:].min()), 2), round(float(atrp), 5)]


def senal_verde(df, ruedas=5):
    """Devuelve (techo de las últimas 20 ruedas con la de hoy, señal) con la señal más reciente de las últimas `ruedas` ruedas, o None.
    señal = [fecha ISO, strong 0/1, cierre, techo roto, zona máx., zona mín., Z-score, ruedas atrás (0 = hoy), fecha de cierre (rueda 20)]."""
    x = df[["High", "Low", "Close", "Volume"]].copy()
    x = x.dropna(subset=["High", "Low", "Close"])
    if len(x) < 205:
        return None
    c, h = x["Close"], x["High"]
    techo20 = round(float(h.iloc[-20:].max()), 2)
    techo = h.shift(1).rolling(20).max()                   # máximo de las 20 ruedas anteriores
    mom = c.diff()
    z = (mom - mom.rolling(50).mean()) / mom.rolling(50).std(ddof=0)
    ruptura = (c > techo) & (c.shift(1) <= techo.shift(1)) & (z > 1.5)
    verde = ruptura & (c > c.rolling(200).mean())
    vol = x["Volume"]
    strong = verde & (z > 2.5) & (vol > 1.5 * vol.rolling(20).mean())
    for atras in range(ruedas):
        i = len(x) - 1 - atras
        if bool(verde.iloc[i]):
            fs = pd.Timestamp(x.index[i]).date().isoformat()
            return techo20, [fs, int(bool(strong.iloc[i])), round(float(c.iloc[i]), 2), round(float(techo.iloc[i]), 2),
                             round(float(h.iloc[i - 1]), 2), round(float(x["Low"].iloc[i - 1]), 2), round(float(z.iloc[i]), 2), atras, fecha_cierre(fs)]
    return techo20, None


def bajar_ccl(previo):
    """Dólar contado con liquidación (dolarapi.com). Si falla, se conserva el valor anterior (con su fecha)."""
    try:
        import urllib.request
        req = urllib.request.Request("https://dolarapi.com/v1/dolares/contadoconliqui", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.load(r)
        v = float(j["venta"]); c = float(j.get("compra") or v)
        if v > 0:
            return {"c": c, "v": v, "f": j.get("fechaActualizacion") or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "fuente": "dolarapi.com"}
    except Exception as e:
        print("CCL no disponible:", e, file=sys.stderr)
    return previo


def bajar_cedears(acciones, previo):
    """Último cierre en pesos del CEDEAR de cada acción (Yahoo, ticker con sufijo .BA): {símbolo: [precio, fecha]}. Si no se consigue, se conserva lo anterior solo si tiene 10 días o menos."""
    mapa = {s: s.replace("-", "").replace(".", "") + ".BA" for s in acciones}
    inv = {v: k for k, v in mapa.items()}; tk = list(inv); out = {}
    for i in range(0, len(tk), LOTE):
        lote = tk[i:i + LOTE]
        try:
            d = yf.download(lote, period="10d", interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
            for t in lote:
                try:
                    x = (d[t] if isinstance(d.columns, pd.MultiIndex) else d)["Close"].dropna()
                    if len(x) and float(x.iloc[-1]) > 0:
                        out[inv[t]] = [round(float(x.iloc[-1]), 2), pd.Timestamp(x.index[-1]).strftime("%Y-%m-%d")]
                except Exception:
                    pass
        except Exception as e:
            print("lote de CEDEARs falló:", e, file=sys.stderr)
    hoy = dt.date.today()
    for s, v in (previo or {}).items():
        if s not in out:
            try:
                if (hoy - dt.date.fromisoformat(v[1])).days <= 10: out[s] = v
            except Exception:
                pass
    return out


def limpiar(s, x):
    """Deja solo Open y Close completos. A veces Yahoo manda la vela del último día con el cierre vacío (todavía sin consolidar)
    aunque la rueda ya terminó: en ese caso se toma el último precio de ese día de las velas de 5 minutos."""
    x = x[["Open", "High", "Low", "Close", "Volume"]]
    try:
        if len(x) and pd.isna(x["Close"].iloc[-1]):
            dia = pd.Timestamp(x.index[-1]).date(); ahora = dt.datetime.now(dt.timezone.utc)
            if dia < ahora.date() or ahora.hour >= 21:
                h = yf.Ticker(s).history(period="5d", interval="5m", auto_adjust=False)
                h = h[[k.date() == dia for k in h.index]].dropna(subset=["Close"])
                if len(h):
                    x = x.copy(); x.iloc[-1, x.columns.get_loc("Close")] = float(h["Close"].iloc[-1])
                    if pd.isna(x["Open"].iloc[-1]): x.iloc[-1, x.columns.get_loc("Open")] = float(h["Open"].iloc[0])
                    if pd.isna(x["Volume"].iloc[-1]): x.iloc[-1, x.columns.get_loc("Volume")] = float(h["Volume"].sum())
                    print(s, "cierre del", dia, "completado con el último precio del día:", round(float(h["Close"].iloc[-1]), 2))
    except Exception as e:
        print(s, "no se pudo completar el cierre:", e, file=sys.stderr)
    return x.dropna(subset=["Open", "High", "Low", "Close"])


def bajar_lote(simbolos):
    """Devuelve {símbolo: DataFrame} con lo que se pudo bajar."""
    out = {}
    try:
        d = yf.download(simbolos, start=INICIO, interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
        for s in simbolos:
            try:
                x = d[s] if isinstance(d.columns, pd.MultiIndex) else d
                x = limpiar(s, x)
                if len(x) > 30:
                    out[s] = x
            except Exception:
                pass
    except Exception as e:
        print("lote falló:", e, file=sys.stderr)
    return out


def bajar_uno(s):
    for intento in range(3):
        try:
            x = yf.Ticker(s).history(start=INICIO, interval="1d", auto_adjust=False)
            x = limpiar(s, x)
            if len(x) > 30:
                x.index = x.index.tz_localize(None) if x.index.tz is not None else x.index
                return x
        except Exception as e:
            print(s, "intento", intento + 1, e, file=sys.stderr)
        time.sleep(2 + 3 * intento)
    return None


def main():
    cfg = json.load(open(RUTA_TICKERS))
    favoritas = json.load(open("data/favoritas.json"))
    filtro = cfg["filtro"]
    acciones = [a for a in dict.fromkeys(favoritas + cfg.get("extras", [])) if a != filtro]
    simbolos = [filtro] + acciones
    previo, previo_atr, previo_ccl, previo_ced = {}, {}, None, {}
    if os.path.exists(RUTA_SALIDA):
        try:
            _p = json.load(open(RUTA_SALIDA)); previo, previo_atr, previo_ccl, previo_ced = _p.get("t", {}), _p.get("atr", {}), _p.get("ccl"), _p.get("cedear", {})
        except Exception:
            previo = {}
    datos = {}
    for i in range(0, len(simbolos), LOTE):
        datos.update(bajar_lote(simbolos[i:i + LOTE]))
    for s in simbolos:
        if s not in datos:
            x = bajar_uno(s)
            if x is not None:
                datos[s] = x
    # A veces la descarga en lote deja algunos símbolos un día atrasados: se vuelven a pedir de a uno.
    if datos:
        tope = max(pd.Timestamp(x.index[-1]) for x in datos.values())
        atrasados = [s for s, x in datos.items() if pd.Timestamp(x.index[-1]) < tope]
        for s in atrasados:
            x = bajar_uno(s)
            if x is not None and pd.Timestamp(x.index[-1]) > pd.Timestamp(datos[s].index[-1]):
                datos[s] = x
        print(f"{len(atrasados)} símbolos atrasados, {sum(1 for s in atrasados if pd.Timestamp(datos[s].index[-1]) >= tope)} corregidos")
    if filtro not in datos:
        sys.exit(f"No se pudo bajar {filtro}: no se guarda nada.")
    # Sin el pico de hoy: si el mercado todavía está abierto, la última vela no está completa.
    ahora = dt.datetime.now(dt.timezone.utc)
    t, chk, fallas, atr, tec, sen = {}, {}, [], {}, {}, {}
    for s in simbolos:
        if s in datos:
            x = datos[s]
            ult = pd.Timestamp(x.index[-1]).date()
            if ult == ahora.date() and ahora.hour < 21:
                x = x.iloc[:-1]
            t[s], n, d, c = codificar(x)
            chk[s] = [n, d, c]
            try:
                sv = senal_verde(x)
                if sv:
                    tec[s] = sv[0]
                    if sv[1]: sen[s] = sv[1]
            except Exception as e:
                print(s, "señal no calculada:", e, file=sys.stderr)
            v = volatilidad(x)
            if v: atr[s] = v
            elif s in previo_atr: atr[s] = previo_atr[s]
        elif s in previo:
            t[s] = previo[s]; fallas.append(s + " (se conservó lo anterior)")
            if s in previo_atr: atr[s] = previo_atr[s]
        else:
            fallas.append(s + " (sin datos)")
    ultimo_dia = max(v[1] for v in chk.values())
    vix = []
    try:   # miedo del mercado: cierre y máximo del día de las últimas ruedas (si falla, el resto se guarda igual)
        h = yf.Ticker("^VIX").history(period="1mo", interval="1d", auto_adjust=False).dropna(subset=["Close", "High"]).tail(10)
        vix = [[i.strftime("%Y-%m-%d"), round(float(r["Close"]), 2), round(float(r["High"]), 2)] for i, r in h.iterrows()]
    except Exception as e:
        print("VIX no disponible:", e, file=sys.stderr)
    ccl = bajar_ccl(previo_ccl)
    try:
        cedear = bajar_cedears([a for a in acciones if a in t], previo_ced)
    except Exception as e:
        print("CEDEARs no disponibles:", e, file=sys.stderr); cedear = previo_ced
    salida = {"gen": ahora.strftime("%Y-%m-%dT%H:%M:%SZ"), "filtro": filtro, "ultimo": (EPOCH + dt.timedelta(days=ultimo_dia)).isoformat(), "def": [x for x in cfg.get("defensivas", []) if x in t], "vix": vix, "atr": atr, "ccl": ccl, "cedear": cedear, "tec": tec, "sen": sen, "t": t}
    with open(RUTA_SALIDA, "w") as f:
        json.dump(salida, f, separators=(",", ":"))
        f.write("\n")
    print(f"Círculos verdes en las últimas 5 ruedas: {len(sen)} ({', '.join(k + ' ' + v[0] + (' strong' if v[1] else '') for k, v in sen.items()) or 'ninguno'})")
    print(f"{len(t)} símbolos guardados, último día {salida['ultimo']}, {os.path.getsize(RUTA_SALIDA) // 1024} KB. Problemas: {fallas or 'ninguno'}")


if __name__ == "__main__":
    main()
