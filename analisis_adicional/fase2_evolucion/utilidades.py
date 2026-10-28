"""
utilidades.py — Fase 2: rutas, carga de artefactos (Marco y Fase 2), plan de entrenamientos y manifiesto.

Sólo depende de NumPy y de analisis_adicional/comun.py (Fase 1, no se modifica). Puede importarse tanto desde el
entorno virtual de entrenamiento (con gymnasium) como desde el Python del sistema (análisis y figuras).

Formato de los artefactos NUEVOS: .npz sin pickle (np.load(..., allow_pickle=False)); la metadata va en JSON.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

F2 = Path(__file__).resolve().parent
RAIZ = F2.parent.parent
FASE1 = RAIZ / "analisis_adicional"
ART = F2 / "artefactos"
RES = F2 / "resultados"
FIG = F2 / "figuras"
sys.path.insert(0, str(FASE1))
import comun as c1  # noqa: E402  (Fase 1: carga restringida de pickles, estadística)

GENERADORES = c1.GENERADORES
ALGORITMOS = c1.ALGORITMOS
SEMILLAS = list(range(5))
SEED_VAL, N_VAL, SEED_TEST, N_TEST = c1.SEED_VAL, c1.N_VAL, c1.SEED_TEST, c1.N_TEST

# Valores por defecto de trader_rl.Config() (se comprueba en entrenar.py que coinciden con el código real)
DEFAULT = dict(algoritmo="qlearning", n_bins=10, discretizacion="cuantiles", gamma=0.95, alpha=0.05, alpha_decay=1.0,
               alpha_min=0.001, eps_inicial=1.0, eps_final=0.01, frac_exploracion=0.6, n_episodios=2000, n_steps=500)
GEN_SENAL_DEBIL = ["reversion", "regimenes"]   # generadores prioritarios (recorte de alcance declarado)


def asegurar():
    for d in (ART, RES, FIG):
        d.mkdir(exist_ok=True)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


SUFIJOS_TEXTO = {".py", ".md", ".csv", ".json", ".txt", ".log", ".ipynb"}


def sha256_lf(ruta):
    """sha256 con saltos de línea normalizados a LF en archivos de texto. Sirve para comprobar archivos en otro sistema
    operativo (Git con core.autocrlf cambia CRLF/LF al hacer checkout). Los binarios (pkl, png, npz) se hashean tal cual."""
    ruta = Path(ruta)
    datos = ruta.read_bytes()
    if ruta.suffix.lower() in SUFIJOS_TEXTO:
        datos = datos.replace(b"\r\n", b"\n")
    return hashlib.sha256(datos).hexdigest()


def archivos_protegidos():
    """Alcance de la INSTANTÁNEA DE INICIO de la Fase 2 (código de Marco, bitácora, notebook, resultados y figuras originales
    más todo lo que había en analisis_adicional/ salvo fase2_evolucion/). Sólo la usa manifiesto.py --huellas, que se niega a
    sobrescribir la instantánea. La comprobación actual (manifiesto.py --comprobar) NO recorre carpetas: compara contra listas
    declaradas en manifiesto_huellas_inicio.json y manifiesto_integridad.json."""
    lista = [RAIZ / n for n in ("trader_rl.py", "experimentos.py", "graficas.py", "BITACORA.md", "taller_trader.ipynb")]
    lista += sorted((RAIZ / "resultados").glob("*")) + sorted((RAIZ / "figuras").glob("*"))
    lista += sorted(p for p in FASE1.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "fase2_evolucion" not in p.parts)
    return lista


def huellas_protegidas():
    return {str(p.relative_to(RAIZ)).replace("\\", "/"): sha256(p) for p in archivos_protegidos()}


# -----------------------------------------------------------------------------
# Artefactos de Marco
# -----------------------------------------------------------------------------
def marco_etapas():
    return {e: c1.cargar_pickle(e) for e in ("alpha", "gamma", "epsilon", "discretizacion")}


def cadena_marco(etapas):
    """(gen, algo) -> {'E1': runs alpha-ganadora, 'E2': runs gamma-ganadora, 'E3': runs epsilon-ganadora} (5 semillas cada una)."""
    g = {e: c1.ganadores(etapas[e]) for e in ("alpha", "gamma", "epsilon")}
    return {k: {"E1": g["alpha"][k], "E2": g["gamma"][k], "E3": g["epsilon"][k]} for k in g["epsilon"]}


def id_run(grupo, gen, algo, seed):
    return f"{grupo}|{gen}|{algo}|s{seed}"


def tarea(grupo, gen, algo, seed, cfg, descripcion):
    cfg = dict(cfg)
    cfg["algoritmo"] = algo
    return dict(id=id_run(grupo, gen, algo, seed), grupo=grupo, generador=gen, algoritmo=algo, seed=seed, cfg=cfg,
                descripcion=descripcion)


def plan_entrenamientos(etapas):
    """Todos los entrenamientos NUEVOS de la Fase 2, por grupo. Cada grupo es un .npz."""
    cadena = cadena_marco(etapas)
    final = {k: dict(v["E3"][0]["cfg"]) for k, v in cadena.items()}
    grupos = {}

    def add(nombre, gens, cfg_fn, desc):
        grupos[nombre] = [tarea(nombre, g, a, s, cfg_fn(g, a), desc) for g in gens for a in ALGORITMOS for s in SEMILLAS]

    # E0 — configuración inicial (Config() por defecto) con distintos N. N=5000 equivale a "alpha=0.05 constante, 5000 ep".
    for n in (500, 1000, 2000, 5000):
        add(f"E0_N{n}", GENERADORES, lambda g, a, n=n: {**DEFAULT, "n_episodios": n},
            f"E0: Config() por defecto (alpha=0.05 constante, gamma=0.95, eps 1->0.01 f=0.6, 10 cuantiles), N={n}")
    add("E0_N10000", GEN_SENAL_DEBIL, lambda g, a: {**DEFAULT, "n_episodios": 10000},
        "E0 con N=10000 (sólo reversión y regímenes: recorte de alcance)")
    # Configuración final de Marco (ganadora de la etapa epsilon) con N distinto del de 5000 ya entrenado
    for n in (500, 1000, 2000):
        add(f"F_N{n}", GENERADORES, lambda g, a, n=n: {**final[(g, a)], "n_episodios": n},
            f"Config final de Marco (E3) reentrenada con N={n}")
    add("F_N10000", GEN_SENAL_DEBIL, lambda g, a: {**final[(g, a)], "n_episodios": 10000},
        "Config final de Marco (E3) con N=10000 (sólo reversión y regímenes)")
    # alpha x N sobre la configuración base de Marco (gamma=0.95, eps f=0.6, 10 cuantiles); sólo mercados de señal débil
    for alpha in (0.002, 0.005):
        for n in (500, 1000, 2000, 10000):
            tag = f"alfa{alpha}_N{n}".replace(".", "p")
            add(tag, GEN_SENAL_DEBIL, lambda g, a, alpha=alpha, n=n: {**DEFAULT, "alpha": alpha, "n_episodios": n},
                f"alpha={alpha} constante sobre la base de Marco, N={n}")
    # Intervención para contrastar el mecanismo de bloqueo: E0 pero con epsilon constante 0.1 (la acción "mantener dentro" se sigue revisitando)
    add("E0eps0p1_N5000", GEN_SENAL_DEBIL, lambda g, a: {**DEFAULT, "eps_inicial": 0.1, "eps_final": 0.1, "n_episodios": 5000},
        "E0 (alpha=0.05) con epsilon constante 0.1, N=5000: intervención sobre el bloqueo (sólo reversión y regímenes)")
    # Duplicado INSTRUMENTADO de corridas de Marco (alpha-stage, N=5000): verifica la reproducción y aporta visitas por (estado, acción)
    rep = []
    for run in etapas["alpha"]:
        if run["generador"] in GEN_SENAL_DEBIL and run["variante"] in ("α=0.002", "α=0.005"):
            rep.append(tarea("REP", run["generador"], run["algoritmo"], run["seed"], run["cfg"],
                             f"Réplica instrumentada de Marco alpha-stage {run['variante']}"))
            rep[-1]["variante_marco"] = run["variante"]
    grupos["REP"] = rep
    return grupos


def guardar_grupo(nombre, runs, forzar=False):
    asegurar()
    ruta = ART / f"{nombre}.npz"
    if ruta.exists() and not forzar:
        raise FileExistsError(f"{ruta} ya existe; usa --forzar para sobrescribir un artefacto de Fase 2 (los de Marco nunca se tocan)")
    datos, meta = {}, []
    for i, r in enumerate(runs):
        p = f"r{i:04d}_"
        datos[p + "Q"] = r["Q"]
        datos[p + "cortes"] = r["cortes"]
        for k, v in r["hist"].items():
            datos[p + "h_" + k] = np.asarray(v, dtype=np.float32)
        for k in ("balance_final", "logret", "operaciones"):
            datos[p + "v_" + k] = np.asarray(r["val"][k])
            datos[p + "t_" + k] = np.asarray(r["test"][k])
        datos[p + "vis_total"] = r["visitas_total"]
        datos[p + "vis_tarde"] = r["visitas_tarde"]
        meta.append({k: r[k] for k in ("id", "grupo", "generador", "algoritmo", "seed", "cfg", "tiempo")})
    datos["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(ruta, **datos)
    return ruta


def cargar_grupo(nombre):
    """Devuelve lista de dicts con la misma estructura que los de Marco (cfg, seed, Q, cortes, hist, val, test, tiempo...)."""
    with np.load(ART / f"{nombre}.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        out = []
        for i, m in enumerate(meta):
            p = f"r{i:04d}_"
            r = dict(m)
            r["Q"], r["cortes"] = z[p + "Q"], z[p + "cortes"]
            r["hist"] = {k[len(p) + 2:]: z[k] for k in z.files if k.startswith(p + "h_")}
            r["val"] = {k: z[p + "v_" + k] for k in ("balance_final", "logret", "operaciones")}
            r["test"] = {k: z[p + "t_" + k] for k in ("balance_final", "logret", "operaciones")}
            r["visitas_total"], r["visitas_tarde"] = z[p + "vis_total"], z[p + "vis_tarde"]
            out.append(r)
    return out


def grupos_existentes():
    return sorted(p.stem for p in ART.glob("*.npz"))


def registrar(texto):
    asegurar()
    with open(F2 / "registro_ejecucion.log", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {texto}\n")
