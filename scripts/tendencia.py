"""Baja de Yahoo Finance las velas diarias (apertura y cierre, ajustadas por splits) de SPY y de las acciones de data/tendencia_tickers.json
y las guarda en data/tendencia_acciones.json para el dashboard de tendencia (tendencia/index.html).

Formato compacto: por cada símbolo, un texto con una vela por tramo separado por coma: "día.apertura.cierre", todo en base 36.
 - día: el primero es el día absoluto (días desde 1970); los demás, la diferencia con la vela anterior.
 - apertura: diferencia, en centavos, con el cierre anterior (la primera, el valor absoluto), en zigzag (negativos impares).
 - cierre: diferencia, en centavos, con la apertura de esa vela, en zigzag.
Si un símbolo falla, se conserva lo que ya estaba guardado de ese símbolo. Si falla SPY (el filtro de mercado) el script termina con error y no se guarda nada."""
import json, os, sys, time, datetime as dt
import pandas as pd
import yfinance as yf

INICIO = "2023-01-01"      # hacen falta ~200 días antes de enero de 2024 para la media del permiso
RUTA_TICKERS = "data/tendencia_tickers.json"
RUTA_SALIDA = "data/tendencia_acciones.json"
LOTE = 20
EPOCH = dt.date(1970, 1, 1)


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


def bajar_lote(simbolos):
    """Devuelve {símbolo: DataFrame} con lo que se pudo bajar."""
    out = {}
    try:
        d = yf.download(simbolos, start=INICIO, interval="1d", auto_adjust=False, progress=False, group_by="ticker", threads=True)
        for s in simbolos:
            try:
                x = d[s] if isinstance(d.columns, pd.MultiIndex) else d
                x = x[["Open", "Close"]].dropna()
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
            x = yf.Ticker(s).history(start=INICIO, interval="1d", auto_adjust=False)[["Open", "Close"]].dropna()
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
    previo = {}
    if os.path.exists(RUTA_SALIDA):
        try:
            previo = json.load(open(RUTA_SALIDA)).get("t", {})
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
    if filtro not in datos:
        sys.exit(f"No se pudo bajar {filtro}: no se guarda nada.")
    # Sin el pico de hoy: si el mercado todavía está abierto, la última vela no está completa.
    ahora = dt.datetime.now(dt.timezone.utc)
    t, chk, fallas = {}, {}, []
    for s in simbolos:
        if s in datos:
            x = datos[s]
            ult = pd.Timestamp(x.index[-1]).date()
            if ult == ahora.date() and ahora.hour < 21:
                x = x.iloc[:-1]
            t[s], n, d, c = codificar(x)
            chk[s] = [n, d, c]
        elif s in previo:
            t[s] = previo[s]; fallas.append(s + " (se conservó lo anterior)")
        else:
            fallas.append(s + " (sin datos)")
    ultimo_dia = max(v[1] for v in chk.values())
    salida = {"gen": ahora.strftime("%Y-%m-%dT%H:%M:%SZ"), "filtro": filtro, "ultimo": (EPOCH + dt.timedelta(days=ultimo_dia)).isoformat(), "def": [x for x in cfg.get("defensivas", []) if x in t], "t": t}
    with open(RUTA_SALIDA, "w") as f:
        json.dump(salida, f, separators=(",", ":"))
        f.write("\n")
    print(f"{len(t)} símbolos guardados, último día {salida['ultimo']}, {os.path.getsize(RUTA_SALIDA) // 1024} KB. Problemas: {fallas or 'ninguno'}")


if __name__ == "__main__":
    main()
