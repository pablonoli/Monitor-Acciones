"""Baja fundamentales de Yahoo Finance (yfinance) para el universo y arma data/datos.json.
Uso: python scripts/fetch.py [cantidad_maxima]"""
import json, sys, time, math, datetime as dt
import yfinance as yf

UNI = json.load(open("data/universe.json"))["list"]
try:
    FALLBACK = json.load(open("data/fallback.json"))
except Exception:
    FALLBACK = {}
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(UNI)
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

def one(u):
    t = u["t"]
    tk = yf.Ticker(t)
    info = tk.info or {}
    price = num(info.get("currentPrice") or info.get("regularMarketPrice"))
    mcap = num(info.get("marketCap"))
    if not price or not mcap:
        return {"t": t, "status": "nodata", "note": "Sin datos en Yahoo"}
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
            if r["status"] != "ok" and u["t"] in FALLBACK:   # sin datos en Yahoo: se usa la última carga de StockAnalysis
                r = dict(FALLBACK[u["t"]], histNote=(FALLBACK[u["t"]].get("histNote") or "") + " · dato de StockAnalysis (27/10/26), no se actualiza")
            out[u["t"]] = r
        time.sleep(0.4)
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
