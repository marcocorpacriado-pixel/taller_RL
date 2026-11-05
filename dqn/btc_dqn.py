"""
DQN sobre BTC-USD, con el mismo corte cronológico que datos_reales.py.

No toca el ajuste de los mercados sintéticos: escribe dqn/resultados/btc.pkl.

Uso, desde este directorio (dqn/):
    python btc_dqn.py              entrena el grid y guarda el pickle
    python btc_dqn.py tabla        sólo la tabla, si el pickle ya existe

Hace falta datos/btc_usd.csv (ya congelado en el repo) y el mismo entorno
Python que el resto del DQN: torch, numpy, pandas, gymnasium.
"""
import os
import pickle
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from datos_reales import CSV, PERIODOS, generador_ventanas, logret, metricas, tramo
from dqn_agente import ConfigDQN, entrenar
from experimentos_dqn import SEMILLA_LISTA
from trader_rl import ejecutar_episodio, politica_aleatoria, politica_buy_and_hold

DIR = Path(__file__).resolve().parent / "resultados"
PKL = DIR / "btc.pkl"
# Mismas tasas que la etapa lr del sintético, y los tres γ del experimento tabular.
GRID = [dict(lr=lr, gamma=g) for lr in (1e-4, 3e-4, 1e-3, 3e-3) for g in (0.9, 0.95, 0.99)]
SEMILLAS_ALEATORIO = [0, 1, 2, 3, 4]


def guardar(datos):
    DIR.mkdir(parents=True, exist_ok=True)
    with open(PKL, "wb") as f:
        pickle.dump(datos, f)


def cargar():
    with open(PKL, "rb") as f:
        return pickle.load(f)


def entrenar_tarea(tarea):
    cambios, seed = tarea
    cfg = replace(ConfigDQN(), **cambios)
    gen = generador_ventanas(tramo("train").values)
    val = tramo("val").values
    test = tramo("test").values
    etiqueta = f"[btc] lr={cfg.lr:g} γ={cfg.gamma} semilla {seed}"
    print(f"{etiqueta}  empieza", flush=True)
    t0 = time.perf_counter()
    agente, hist = entrenar(cfg, gen, seed, series_validacion=[val], eval_cada=250, progreso=etiqueta)
    politica = agente.politica_greedy()
    pesos = {k: v.detach().cpu().clone() for k, v in agente.q.state_dict().items()}
    return {
        "cfg": asdict(cfg), "seed": seed, "pesos": pesos, "escala": agente.escala,
        "tiempo": time.perf_counter() - t0,
        "hist": {k: np.asarray(v, dtype=np.float32) for k, v in hist.items()},
        "val": ejecutar_episodio(val, politica),
        "test": ejecutar_episodio(test, politica),
    }


def baselines():
    """Buy & Hold una vez; aleatorio con 5 semillas, como en el PDF."""
    return {p: {"buy&hold": [ejecutar_episodio(tramo(p).values, politica_buy_and_hold)],
                "aleatorio": [ejecutar_episodio(tramo(p).values, politica_aleatoria(s))
                              for s in SEMILLAS_ALEATORIO]}
            for p in ("val", "test")}


def ejecutar():
    if not Path(CSV).exists():
        raise SystemExit(f"No está {CSV}. Hace falta la serie congelada del repo.")
    tareas = [(g, s) for g in GRID for s in SEMILLA_LISTA]
    n = min(len(tareas), os.cpu_count() or 1)
    print(f"[btc] {len(tareas)} entrenamientos, {n} a la vez. "
          f"Una línea por entrenamiento más o menos cada minuto.", flush=True)
    print(f"[btc] periodos {PERIODOS['train'][0]}→{PERIODOS['train'][1]} / "
          f"{PERIODOS['val'][0]}→{PERIODOS['val'][1]} / "
          f"{PERIODOS['test'][0]}→{PERIODOS['test'][1]}", flush=True)
    t0 = time.perf_counter()
    resultados = []
    with get_context("spawn").Pool(n) as pool:
        for r in pool.imap_unordered(entrenar_tarea, tareas):
            resultados.append(r)
            print(f"[btc] {len(resultados)}/{len(tareas)} terminados  "
                  f"lr={r['cfg']['lr']:g} γ={r['cfg']['gamma']} semilla {r['seed']}  "
                  f"{r['tiempo'] / 60:.1f} min  val {logret(r['val']) * 100:+.1f}%", flush=True)
    datos = {"resultados": resultados, "baselines": baselines()}
    guardar(datos)
    print(f"[btc] guardado en {PKL}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    tabla(datos)


def seleccion(resultados):
    """Las semillas de la pareja (lr, γ) con mejor log-retorno medio en validación."""
    grupos = {}
    for r in resultados:
        c = r["cfg"]
        grupos.setdefault((c["lr"], c["gamma"]), []).append(r)
    mejor, elegido = None, None
    for clave, runs in grupos.items():
        puntuacion = np.mean([logret(r["val"]) for r in runs])
        if mejor is None or puntuacion > mejor:
            mejor, elegido = puntuacion, runs
    return elegido


def _celda(episodios, clave):
    v = np.array([metricas(ep)[clave] for ep in episodios], dtype=float)
    fmt = "{:.2f}" if clave == "Sharpe" else "{:.1f}"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}"


def tabla(datos=None):
    datos = datos if datos is not None else cargar()
    elegidos = seleccion(datos["resultados"])
    c = elegidos[0]["cfg"]
    claves = list(metricas(datos["baselines"]["test"]["buy&hold"][0]))
    lineas = [f"DQN elegido en validación: lr={c['lr']:g}, γ={c['gamma']}, "
              f"{len(elegidos)} semillas.", ""]
    for periodo in ("val", "test"):
        ini, fin = PERIODOS[periodo]
        lineas += [f"{'Validación' if periodo == 'val' else 'Test'} ({ini} a {fin})",
                   "| Agente | " + " | ".join(claves) + " |",
                   "|" + "---|" * (len(claves) + 1)]
        filas = [("Buy & Hold", datos["baselines"][periodo]["buy&hold"]),
                 ("Aleatorio", datos["baselines"][periodo]["aleatorio"]),
                 ("DQN", [r[periodo] for r in elegidos])]
        for nombre, episodios in filas:
            lineas.append(f"| {nombre} | " + " | ".join(_celda(episodios, k) for k in claves) + " |")
        lineas.append("")
    texto = "\n".join(lineas)
    print(texto)
    return texto


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tabla":
        tabla()
    else:
        ejecutar()
