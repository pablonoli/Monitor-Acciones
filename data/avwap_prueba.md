# Prueba del AVWAP

Acciones: AAPL, MSFT, NVDA, MELI, KO, WMT, JPM, XOM, DIS, PFE. Datos de 10 años, una señal por racha (separadas por al menos 10 ruedas). Retornos a 20 y 60 ruedas, en %.

| Regla | Señales | Prom. 20r | Mediana 20r | % positivas 20r | Prom. 60r | % positivas 60r | Aguantó/frenó |
|---|---|---|---|---|---|---|---|
| Soporte: AVWAP desde el mínimo de 52 semanas (precio tocándolo desde arriba) | 205 | 1.23 | 0.96 | 57.6 | 4.72 | 64.3 | 51.2 |
| Control: la misma regla con la media de 200 ruedas | 139 | 2.72 | 2.12 | 66.9 | 6.23 | 67.4 | 62.6 |
| Resistencia: AVWAP desde el máximo de 52 semanas (precio tocándolo desde abajo) | 115 | 1.3 | 0.9 | 55.8 | 4.37 | 62.7 | 29.2 |
| Ruptura: el precio supera ese AVWAP en más de 2% | 92 | 0.6 | 0.01 | 50.5 | 5.43 | 61.8 | 54.9 |
| Base: cualquier rueda | 22420 | 1.6 | 1.3 | 57.6 | 4.75 | 62.1 | — |

## Por acción (señales de soporte AVWAP / control media 200 / resistencia)

| Acción | Soporte AVWAP (n, prom 20r, % pos) | Media 200 (n, prom 20r, % pos) | Resistencia (n, prom 20r, % pos) | Base prom 20r |
|---|---|---|---|---|
| AAPL | 18, 2.75, 72.2 | 19, 3.04, 73.7 | 12, 3.77, 66.7 | 2.18 |
| MSFT | 16, -1.08, 43.8 | 10, -0.69, 60.0 | 8, -2.26, 50.0 | 1.9 |
| NVDA | 15, 6.67, 64.3 | 13, 5.17, 61.5 | 8, 10.48, 87.5 | 4.27 |
| MELI | 21, 4.81, 60.0 | 23, 5.59, 73.9 | 16, -3.15, 26.7 | 2.6 |
| KO | 33, 1.58, 72.7 | 14, 0.24, 64.3 | 7, 4.38, 85.7 | 0.74 |
| WMT | 18, 0.25, 44.4 | 17, 2.45, 64.7 | 8, 1.64, 71.4 | 1.35 |
| JPM | 23, -0.22, 43.5 | 17, 0.46, 52.9 | 8, -0.94, 50.0 | 1.35 |
| XOM | 18, 0.2, 55.6 | 9, 5.0, 77.8 | 14, 0.59, 42.9 | 1.04 |
| DIS | 19, -3.54, 42.1 | 7, 0.71, 57.1 | 18, 1.27, 66.7 | 0.42 |
| PFE | 24, 1.64, 66.7 | 10, 2.87, 80.0 | 16, 1.07, 43.8 | 0.11 |
