"""Investigación (no es parte del dashboard): vender una parte a +20%, por tipo de acción (defensivas/consumo vs tecnológicas)."""
import time, json
from investigacion import cargar, DEF
import investigacion_fraccion as F

CONSUMO = set(DEF) | set("TJX LOW HD".split())
TECNO = set("NVDA AAPL MSFT AMZN GOOGL AVGO META TSLA ORCL MU NFLX AMD PLTR CSCO LRCX AMAT INTC CRM QCOM TXN KLAC ADBE NOW PANW ANET INTU ADI APH IBM APP ACN ASML TSM SHOP CRWD ARM MRVL".split())

if __name__ == "__main__":
    t0 = time.time(); data = cargar(); print(len(data), "símbolos en", round(time.time() - t0), "s", flush=True)
    F.CASOS[:] = [c for c in F.CASOS if c[0] in ("Actual: todo al cruce", "Vende 25% a +20%", "Vende 1/3 a +20%", "Vende 50% a +20%", "Vende 75% a +20%")]
    salida = []
    for nom, g in (("DEFENSIVAS Y CONSUMO", CONSUMO), ("TECNOLÓGICAS", TECNO)):
        sub = {s: v for s, v in data.items() if s in g or s == "SPY"}
        filas = F.correr(sub); salida.append(f"######## {nom} ({len(sub) - 1} acciones disponibles)\n" + F.texto(filas))
    txt = "\n\n".join(salida); open("data/investigacion_grupos.txt", "w", encoding="utf-8").write(txt); print(txt)
