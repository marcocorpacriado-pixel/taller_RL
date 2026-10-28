"""
verificar_fase2.py — Fase 2: verificaciones de integridad y reproducibilidad (sin entrenar).

    python -B -X utf8 analisis_adicional/fase2_evolucion/verificar_fase2.py [--sin-v5]      (en Windows: py -3 en lugar de python)

Se puede lanzar desde cualquier carpeta; sólo necesita NumPy. V5 necesita además un intérprete con gymnasium: se busca en la variable
de entorno TALLER_RL_PYTHON, luego en ~/.venvs/taller_rl_fase2 (Scripts/python.exe o bin/python) y por último en el intérprete actual.
Con --sin-v5 la comprobación V5 se omite de forma explícita (queda como OMITIDA en el CSV y el código de salida sigue siendo 0).

Comprueba:
  V1  Cada artefacto nuevo (.npz) se reconstruye: la política greedy leída de Q reproduce val/test guardados
      (log-retorno, balance y operaciones) con el entorno de la Fase 1, ya validado contra el notebook.
  V2  La réplica REP de Marco fue idéntica bit a bit (lee verificacion_reproduccion_marco.csv).
  V3  Series y semillas de evaluación (val 2024/50, test 7777/100) coinciden con las de Marco: Buy & Hold guardado.
  V4  Integridad por categorías contra listas explícitas de archivos y sha256 (ver manifiesto.py):
        V4a originales de Marco + Fase 1 · V4b artefactos de la Fase 2 · V4c código e informes de la Fase 2 · V4d adiciones de las Fases 3-4
        · V4e ningún archivo sin declarar ni cachés de bytecode.
  V5  Reejecutar entrenar.py sobre un grupo existente NO sobrescribe el artefacto ni añade líneas al registro (se salta y avisa).
  V6  Git: ningún archivo rastreado modificado; el HEAD de inicio es ancestro del HEAD actual y ningún archivo fuera de
      analisis_adicional/ ha cambiado desde entonces (vale también después de hacer los commits de la contribución).
Escribe resultados/verificacion_fase2.csv.
"""
import sys

sys.dont_write_bytecode = True

import csv
import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np

import manifiesto as man
import utilidades as u

c1 = u.c1
FILAS = []


def reg(nombre, n, detalle, ok, estado=None):
    estado = estado or ("OK" if ok else "FALLO")
    FILAS.append(dict(comprobacion=nombre, n_elementos=n, detalle=detalle, resultado=estado))
    print(f"[{estado}] {nombre}: n={n}; {detalle}", flush=True)


def python_de_entrenamiento():
    """Intérprete con gymnasium: TALLER_RL_PYTHON, el venv por defecto (Windows o POSIX) o el intérprete actual si tiene gymnasium."""
    venv = Path.home() / ".venvs" / "taller_rl_fase2"
    candidatos = ([Path(os.environ["TALLER_RL_PYTHON"])] if os.environ.get("TALLER_RL_PYTHON") else []) + [venv / "Scripts" / "python.exe", venv / "bin" / "python"]
    for c in candidatos:
        if c.is_file():
            return c
    return Path(sys.executable) if importlib.util.find_spec("gymnasium") else None


# V1 ---------------------------------------------------------------------------------------------
val = {g: c1.generar_series(g, u.N_VAL, u.SEED_VAL) for g in u.GENERADORES}
test = {g: c1.generar_series(g, u.N_TEST, u.SEED_TEST) for g in u.GENERADORES}
n_runs = n_emp = 0
dmax = 0.0
for nombre in u.grupos_existentes():
    for r in u.cargar_grupo(nombre):
        comprar, vender, e0, e1 = c1.politica_desde_Q(r["Q"])
        n_runs += 1
        if e0.any() or e1.any():
            n_emp += 1
            continue
        g = r["generador"]
        for conj, series in (("val", val[g]), ("test", test[g])):
            rec, bal, ops = c1.evaluar_tabla(r["Q"], r["cortes"], series)
            dmax = max(dmax, float(np.abs(np.log(bal / 1000.0) - r[conj]["logret"]).max()), float(np.abs(bal - r[conj]["balance_final"]).max() / 1000),
                       float(np.abs(ops - r[conj]["operaciones"]).max()))
reg("V1 política reconstruida de Q == val/test guardados (artefactos nuevos)", n_runs - n_emp, f"max|dif|={dmax:.3g}; {n_emp} corridas con empates exactos en Q excluidas", dmax <= 1e-12)

# V2 ---------------------------------------------------------------------------------------------
ruta = u.RES / "verificacion_reproduccion_marco.csv"
if ruta.exists():
    filas = list(csv.DictReader(open(ruta, encoding="utf-8")))
    ident = sum(f["identica"] == "True" for f in filas)
    reg("V2 réplica de 40 corridas de Marco (Q, curvas, recompensas, balances, val, test) idéntica bit a bit", len(filas), f"{ident}/{len(filas)} idénticas", ident == len(filas))
else:
    reg("V2 réplica de Marco", 0, "falta verificacion_reproduccion_marco.csv", False)

# V3 ---------------------------------------------------------------------------------------------
base = c1.cargar_pickle("baselines")
d = 0.0
for g in u.GENERADORES:
    bh = np.log(np.array([1000 * 0.999 / p[5] * p[-1] for p in test[g]]) / 1000)
    d = max(d, float(np.abs(bh - base[(g, "buy&hold")][0]["logret"]).max()))
reg("V3 series de test regeneradas (seed 7777, 100) == baselines.pkl de Marco", 400, f"max|dif|={d:.3g}", d <= 1e-12)

# V4 ---------------------------------------------------------------------------------------------
ref = __import__("json").loads(man.HUELLAS.read_text(encoding="utf-8"))
integ = man.evaluar_integridad()


def v4(nombre, categorias):
    r = [integ[c] for c in categorias if c in integ]
    n = sum(x["n"] for x in r)
    mal = sum(len(x["cambiados"]) + len(x["ausentes"]) for x in r)
    eol = sum(x["solo_eol"] for x in r)
    detalle = f"idénticos={sum(x['identicos'] for x in r)}, sólo saltos de línea={eol}, modificados/ausentes={mal}"
    reg(nombre, n, detalle, integ["_manifiesto_integridad"] and len(r) == len(categorias) and n > 0 and mal == 0)


v4("V4a originales de Marco y archivos de la Fase 1 sin cambios (sha256 frente a la instantánea de inicio)", ["original_marco", "fase1"])
v4("V4b artefactos de la Fase 2 (.npz, resultados, figuras, registro) sin cambios", ["fase2_artefactos"])
v4("V4c código e informes de la Fase 2 coinciden con su lista declarada (ediciones de Fase 4 registradas)", ["fase2_codigo"])
v4("V4d documentos y archivos añadidos en las Fases 3-4 coinciden con su lista declarada", ["adiciones_fase3", "adiciones_fase4"])
reg("V4e ningún archivo sin declarar ni cachés de bytecode en analisis_adicional/", 1,
    f"sin declarar={len(integ['no_declarados'])} {integ['no_declarados'][:5]}, cachés={len(integ['caches'])}", not integ["no_declarados"] and not integ["caches"])

# V5 ---------------------------------------------------------------------------------------------
py = python_de_entrenamiento()
if "--sin-v5" in sys.argv:
    reg("V5 reejecutar entrenar.py sobre un grupo existente no sobrescribe el artefacto", 0, "omitida a petición (--sin-v5)", True, estado="OMITIDA")
elif py is not None:
    log = u.F2 / "registro_ejecucion.log"
    antes, antes_log = u.sha256(u.ART / "REP.npz"), u.sha256(log)
    out = subprocess.run([str(py), "-B", "-X", "utf8", str(u.F2 / "entrenar.py"), "--grupos", "REP"], capture_output=True, text=True, cwd=u.RAIZ)
    despues, despues_log = u.sha256(u.ART / "REP.npz"), u.sha256(log)
    reg("V5 reejecutar entrenar.py sobre un grupo existente no sobrescribe el artefacto ni el registro", 1,
        f"intérprete={py.name}; salida: {out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()[-80:]}; "
        f"npz {'igual' if antes == despues else 'CAMBIADO'}, registro {'igual' if antes_log == despues_log else 'CAMBIADO'}",
        antes == despues and antes_log == despues_log and "saltado" in out.stdout)
else:
    reg("V5 no sobrescritura", 0, "no hay intérprete con gymnasium (define TALLER_RL_PYTHON o usa --sin-v5)", False)

# V6 ---------------------------------------------------------------------------------------------
git = lambda *a: subprocess.run(["git", *a], cwd=u.RAIZ, capture_output=True, text=True).stdout.strip()
estado = git("status", "--short", "--untracked-files=no")
ancestro = subprocess.run(["git", "merge-base", "--is-ancestor", ref["head"], "HEAD"], cwd=u.RAIZ, capture_output=True).returncode == 0
fuera = [f for f in git("diff", "--name-only", ref["head"], "HEAD").splitlines() if not f.startswith("analisis_adicional/")]
reg("V6 git: sin rastreados modificados; HEAD de inicio ancestro del actual; nada fuera de analisis_adicional/ cambiado desde entonces", 1,
    f"rama={git('branch', '--show-current')}, HEAD={git('rev-parse', 'HEAD')[:7]} (inicio {ref['head'][:7]}, ancestro={ancestro}), "
    f"rastreados modificados={'ninguno' if not estado else estado}, cambiados fuera de analisis_adicional/ desde el inicio={fuera or 'ninguno'}",
    estado == "" and ancestro and not fuera)

u.asegurar()
with open(u.RES / "verificacion_fase2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(FILAS[0]))
    w.writeheader()
    w.writerows(FILAS)
sys.exit(0 if all(f["resultado"] in ("OK", "OMITIDA") for f in FILAS) else 1)
