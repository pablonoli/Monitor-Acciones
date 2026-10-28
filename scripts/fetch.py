"""Baja fundamentales de Yahoo Finance (yfinance) para el universo y arma data/datos.json.
Uso: python scripts/fetch.py [cantidad_maxima]"""
import json, os, sys, time, math, datetime as dt
import yfinance as yf

UNI = json.load(open("data/universe.json"))["list"]
try:
    FALLBACK = json.load(open("data/fallback.json"))
except Exception:
    FALLBACK = {}
ONLY = [x for x in os.environ.get("TICKERS", "").replace(",", " ").split() if x]
if ONLY:
    UNI = [u for u in UNI if u["t"] in ONLY]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else len(UNI)
TODAY = dt.date.today().isoformat()
_fx = {}

def num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def pos(v):
    v = num(v)
    return v if v is not None and v > 0 else None

def fx(cur):
    """unidades de `cur` por 1 US$"""
    if not cur or cur == "USD":
        return 1.0
    if cur not in _fx:
        try:
            _fx[cur] = num(yf.Ticker(f"{cur}=X").fast_info["lastPrice"])
        except Exception:
            _fx[cur] = None
    return _fx[cur]

def row(df, *names):
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None

# ticker de Yahoo distinto del de la lista (la lista a veces trae el código del CEDEAR o el ticker viejo)
ALT = {"BK": "BNY", "MMC": "MRSH", "ELP": "ELPC", "XROX": "XRX", "TRVV": "TRV"}

FREQ = {12: "Mensual", 4: "Trimestral", 2: "Semestral", 1: "Anual"}

def divblock(tk, info, price):
    """Resumen de dividendos: tasa, payout, último pago, fechas, frecuencia, crecimiento y dividendo anual por año."""
    rate = num(info.get("dividendRate"))
    try:
        s = tk.dividends
    except Exception:
        s = None
    if (s is None or not len(s)) and not rate:
        return None
    out = {"rate": rate, "yld": rate / price if rate and price else None, "payout": num(info.get("payoutRatio"))}
    if s is not None and len(s):
        if getattr(s.index, "tz", None) is not None:
            s = s.copy()
            s.index = s.index.tz_localize(None)
        out["last"] = num(s.iloc[-1])
        out["lastEx"] = s.index[-1].date().isoformat()
        hoy = dt.date.today()
        ann = {}
        for d_, v in s.items():
            ann[d_.year] = ann.get(d_.year, 0) + float(v)
        full = sorted((y, v) for y, v in ann.items() if y < hoy.year)
        out["ann"] = [[y, round(v, 4)] for y, v in full[-8:]]
        cut = dt.datetime.now() - dt.timedelta(days=365)
        recent = s[s.index > cut]
        n = len(recent)
        out["ttm"] = round(float(recent.sum()), 4) if n else 0
        out["freq"] = FREQ.get(n) or (f"{n} pagos por año" if n else None)
        if len(full) >= 6 and full[-1][1] > 0 and full[-6][1] > 0:
            out["g5"] = (full[-1][1] / full[-6][1]) ** (1 / 5) - 1
        yrs = 0
        for i in range(len(full) - 1, 0, -1):
            if full[i][1] > full[i - 1][1] * 1.0001:
                yrs += 1
            else:
                break
        out["yrs"] = yrs
    try:
        ex = num(info.get("exDividendDate"))
        if ex:
            out["exInfo"] = dt.datetime.utcfromtimestamp(ex).date().isoformat()
    except Exception:
        pass
    if rate:
        try:
            cal = tk.calendar or {}
            pay = cal.get("Dividend Date")
            if pay:
                out["pay"] = pay.isoformat() if hasattr(pay, "isoformat") else str(pay)
            exn = cal.get("Ex-Dividend Date")
            if exn:
                out["exNext"] = exn.isoformat() if hasattr(exn, "isoformat") else str(exn)
        except Exception:
            pass
    return out

def one(u):
    t = u["t"]
    tk = yf.Ticker(ALT.get(t, t))
    info = tk.info or {}
    price = num(info.get("currentPrice") or info.get("regularMarketPrice"))
    mcap = num(info.get("marketCap"))
    if not mcap and price:   # algunos ADRs vienen sin capitalización: se estima con las acciones en circulación
        try:
            mcap = num(tk.fast_info["marketCap"])
        except Exception:
            mcap = None
        if not mcap:
            sh = num(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
            mcap = price * sh if sh else None
    if not price or not mcap:
        return {"t": t, "status": "nodata", "note": "Sin datos actuales en Yahoo"}
    cur = info.get("financialCurrency") or "USD"
    f = fx(cur)
    if not f:
        return {"t": t, "status": "nodata", "note": f"Sin tipo de cambio {cur}"}
    rev = num(info.get("totalRevenue"))
    ni = num(info.get("netIncomeToCommon"))
    rev = rev / f if rev is not None else None
    ni = ni / f if ni is not None else None
    rate = num(info.get("dividendRate"))
    dy = rate / price if rate and price else None   # dividendYield de Yahoo viene en % y a veces mal escalado
    doc = {
        "t": t, "name": info.get("shortName") or info.get("longName"), "status": "ok", "src": "YF",
        "price": price, "mcap": mcap, "cur": cur, "fx": f, "rev": rev, "ni": ni,
        "epsCur": pos(info.get("forwardEps")), "epsNext": None,
        "peg": pos(info.get("trailingPegRatio")), "dy": dy,
        "lastDiv": num(info.get("dividendRate")),
        "fundAt": dt.datetime.utcnow().isoformat() + "Z", "priceAt": dt.datetime.utcnow().isoformat() + "Z",
    }
    # historial anual: ingresos, ganancia y EPS de Yahoo; P/E, P/S y yield con el precio al cierre de cada ejercicio
    hist = []
    try:
        inc = tk.income_stmt
        px = tk.history(period="7y", interval="1mo")["Close"]
        shares_now = mcap / price
        divs = tk.dividends
        for col in sorted(inc.columns):
            r_, n_, e_ = row(inc[[col]], "Total Revenue"), row(inc[[col]], "Net Income Common Stockholders", "Net Income"), row(inc[[col]], "Diluted EPS")
            rv = num(r_.iloc[0]) if r_ is not None else None
            nv = num(n_.iloc[0]) if n_ is not None else None
            ev = num(e_.iloc[0]) if e_ is not None else None
            p = px[px.index <= str(col.date()) + " 23:59"] if len(px) else px
            p = num(p.iloc[-1]) if len(p) else None
            sh = shares_now
            if p and rv:
                ps = p * sh / (rv / f)
            else:
                ps = None
            pe = p * sh / (nv / f) if p and nv and nv > 0 else None
            y0 = col - dt.timedelta(days=365)
            d_ = divs[(divs.index.tz_localize(None) > y0) & (divs.index.tz_localize(None) <= col)].sum() if len(divs) else 0
            dyh = (num(d_) / p) if p and d_ else None
            hist.append({"fy": str(col.year), "pe": pos(pe), "peg": None, "ps": pos(ps), "dy": dyh,
                         "eps": (ev / f) if ev is not None else None,
                         "rev": rv / f if rv is not None else None, "ni": nv / f if nv is not None else None})
    except Exception as e:
        doc["histNote"] = f"Sin historial: {type(e).__name__}"
    try:
        dv = divblock(tk, info, price)
        if dv:
            doc["div"] = dv
    except Exception as e:
        doc["divNote"] = f"Sin dividendos: {type(e).__name__}"
    if hist:
        doc["hist"] = hist[-5:]
        doc["histAt"] = doc["fundAt"]
        doc.setdefault("histNote", "Fuente: Yahoo Finance" + ("" if cur == "USD" else f" (convertido de {cur} a US$ al tipo de cambio actual)"))
    return doc

def main():
    out, fails = {}, []
    for u in UNI[:LIMIT]:
        r = None
        for intento in range(2):   # un reintento por si Yahoo falló de pasada
            try:
                r = one(u)
                if r["status"] == "ok":
                    break
            except Exception as e:
                r = None
                err = f"{type(e).__name__}: {e}"[:120]
            time.sleep(2)
        if r is None:
            fails.append((u["t"], err))
        else:
            out[u["t"]] = r
        time.sleep(0.4)
    if ONLY:   # corrida parcial: se mezcla con el archivo existente y se deja un diagnóstico
        previo = json.load(open("data/datos.json"))
        previo["co"].update(out)
        previo["updated"] = dt.datetime.utcnow().isoformat() + "Z"
        json.dump(previo, open("data/datos.json", "w"), ensure_ascii=False)
        diag = ["# Diagnóstico de empresas puntuales", ""]
        for u in UNI:
            d = out.get(u["t"], {})
            linea = f"- {u['t']}: {'OK Yahoo' if d.get('src') == 'YF' else 'sigue sin datos'}"
            if d.get("src") != "YF":
                try:
                    inf = yf.Ticker(u["t"]).info or {}
                    linea += f" · tipo={inf.get('quoteType')} precio={inf.get('currentPrice') or inf.get('regularMarketPrice')} mcap={inf.get('marketCap')} nombre={inf.get('longName') or inf.get('shortName')}"
                    q = yf.Search(u["t"], max_results=4).quotes
                    linea += " · búsqueda=" + ", ".join(f"{x.get('symbol')}({x.get('shortname')})" for x in q)
                except Exception as e:
                    linea += f" · error={type(e).__name__}: {str(e)[:100]}"
            diag.append(linea)
        open("data/diagnostico.md", "w").write("\n".join(diag))
        print("\n".join(diag))
        return
    json.dump({"updated": dt.datetime.utcnow().isoformat() + "Z", "co": out}, open("data/datos.json", "w"), ensure_ascii=False)
    ok = [d for d in out.values() if d["status"] == "ok"]
    campos = ["price", "mcap", "rev", "ni", "epsCur", "peg", "dy", "hist"]
    lines = [f"# Prueba Yahoo Finance ({TODAY})", f"Empresas probadas: {len(out) + len(fails)} · con datos: {len(ok)} · sin datos: {len(out) - len(ok)} · errores: {len(fails)}", "",
             "| Campo | Empresas con valor |", "|---|---|"]
    for c in campos:
        lines.append(f"| {c} | {sum(1 for d in ok if d.get(c))} / {len(ok)} |")
    lines += ["", "| Ticker | Moneda | P/E (mcap/ni) | Años de historial |", "|---|---|---|---|"]
    for d in ok:
        pe = d["mcap"] / d["ni"] if d.get("ni") and d["ni"] > 0 else None
        lines.append(f"| {d['t']} | {d['cur']} | {pe and round(pe, 1)} | {len(d.get('hist', []))} |")
    if fails:
        lines += ["", "Errores:"] + [f"- {t}: {m}" for t, m in fails]
    open("data/reporte.md", "w").write("\n".join(lines))
    print("\n".join(lines))

main()
