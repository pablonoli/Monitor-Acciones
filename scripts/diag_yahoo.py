import yfinance as yf, pandas as pd, datetime as dt
print("ahora UTC:", dt.datetime.now(dt.timezone.utc))
for s in ("MSFT", "AAPL", "SPY"):
    h = yf.Ticker(s).history(period="10d", interval="1d", auto_adjust=False)
    print(s, "history:\n", h[["Open", "High", "Low", "Close", "Volume"]].tail(5))
    d = yf.download(s, period="10d", interval="1d", auto_adjust=False, progress=False)
    print(s, "download:\n", d.tail(4))
    fi = yf.Ticker(s).fast_info
    print(s, "fast_info last_price:", fi.get("last_price"), "previous_close:", fi.get("previous_close"), "last_volume:", fi.get("last_volume"))
