"""Abre un issue en el repositorio cuando aparece un «círculo verde» Strong con ATR % de 2% o más (la regla de operar; los verdes comunes no se avisan) (ver scripts/tendencia.py).
Lee data/tendencia_acciones.json ("sen", "atr", "ultimo") y recuerda lo ya avisado en data/senales_avisadas.json.
 - Una señal se avisa una sola vez. Las de un mismo día van juntas en un solo issue.
 - La primera vez (si el archivo de avisos no existe) las señales de días anteriores se marcan como avisadas sin abrir issue; solo se avisa la del último día.
Usa el GITHUB_TOKEN que trae GitHub Actions (variable GH_TOKEN para la herramienta «gh»). Con DRY_RUN=1 solo muestra lo que haría."""
import json, os, subprocess, sys, tempfile, datetime as dt

DATOS, AVISADAS = "data/tendencia_acciones.json", "data/senales_avisadas.json"


def f(d):
    return dt.date.fromisoformat(d).strftime("%d/%m/%Y")


def num(x, dec=2):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def cuerpo(fecha, filas, atr):
    out = [f"Rupturas alcistas válidas del **{f(fecha)}** (cierre diario sobre el máximo de las 20 ruedas anteriores, con momentum fuerte y por encima de la SMA 200).", "",
           "**Plan:** comprar en la apertura siguiente · stop en el mínimo de 20 ruedas · vender la mitad a +2R y llevar el stop a la entrada · **cerrar todo al cierre de la rueda 20** (columna «Cerrar todo»), aunque no haya tocado ni el stop ni el objetivo.", "",
           "| Acción | Cierre | Strong | Zona (máx. – mín. de la vela anterior) | ATR % | Stop amplio (mín. 20 ruedas) | Cerrar todo (rueda 20) |", "|---|---|---|---|---|---|---|"]
    for s, v in filas:
        a = atr.get(s)
        atrp = f"{num(100 * a[2], 1)}%" if a and len(a) > 2 else "—"
        stop = f"US$ {num(a[1])} (−{num(100 * (1 - a[1] / v[2]), 1)}%)" if a and a[1] and a[1] < v[2] else "—"
        cierre = f"{f(v[8])}" if len(v) > 8 and v[8] else "—"
        out.append(f"| **{s}** | US$ {num(v[2])} | {'Sí' if v[1] else 'No'} | US$ {num(v[4])} – US$ {num(v[5])} | {atrp} | {stop} | {cierre} |")
    out += ["", "Strong = Z-score mayor a 2,5 y volumen mayor a 1,5 veces el promedio de 20 ruedas. El stop amplio es el mínimo de las últimas 20 ruedas al día de la actualización. «Cerrar todo» = 20 ruedas hábiles después de la señal (la de entrada cuenta como la 1), salteando fines de semana y feriados de la bolsa de EE. UU.",
            "Aviso automático de «Tendencia acciones». Es una señal técnica, no una recomendación."]
    return "\n".join(out)


def abrir(titulo, texto):
    if os.environ.get("DRY_RUN"):
        print("=== ISSUE (simulado) ===\n" + titulo + "\n" + texto + "\n")
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as t:
        t.write(texto)
    base = ["gh", "issue", "create", "--title", titulo, "--body-file", t.name]
    dueno = os.environ.get("GITHUB_REPOSITORY_OWNER")
    for extra in ([["--assignee", dueno]] if dueno else []) + [[]]:   # asignado al dueño para que le llegue la notificación; si falla, sin asignar
        r = subprocess.run(base + extra, capture_output=True, text=True)
        if r.returncode == 0:
            print("Issue creado:", r.stdout.strip()); return True
        print("No se pudo crear el issue:", r.stderr.strip(), file=sys.stderr)
    return False


def main():
    d = json.load(open(DATOS))
    sen, atr, ultimo = d.get("sen", {}), d.get("atr", {}), d.get("ultimo", "")
    primera = not os.path.exists(AVISADAS)
    av = set(json.load(open(AVISADAS))) if not primera else set()
    pendientes = {}
    for s, v in sen.items():
        k = f"{s}|{v[0]}"
        if k in av:
            continue
        a = atr.get(s)
        if not (v[1] and a and len(a) > 2 and a[2] >= 0.02):   # solo tu regla: Strong con ATR % de 2% o más
            continue
        if primera and v[0] < ultimo:
            av.add(k); continue
        pendientes.setdefault(v[0], []).append((s, v))
    ok = True
    for fecha in sorted(pendientes):
        filas = sorted(pendientes[fecha], key=lambda x: (-x[1][1], x[0]))
        nombres = [s for s, _ in filas]
        titulo = (f"Círculo verde: {nombres[0]} – {f(fecha)}" if len(nombres) == 1 else f"Círculos verdes: {', '.join(nombres)} – {f(fecha)}")
        if abrir(titulo, cuerpo(fecha, filas, atr)):
            av.update(f"{s}|{fecha}" for s in nombres)
        else:
            ok = False
    corte = (dt.date.today() - dt.timedelta(days=90)).isoformat()
    guardar = sorted(k for k in av if k.split("|")[1] >= corte)
    if not os.environ.get("DRY_RUN"):
        json.dump(guardar, open(AVISADAS, "w"), indent=0); open(AVISADAS, "a").write("\n")
    print(f"Señales en las últimas 5 ruedas: {len(sen)}; avisos nuevos: {sum(len(v) for v in pendientes.values())}.")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
