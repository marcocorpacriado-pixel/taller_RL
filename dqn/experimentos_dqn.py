"""
Ajuste y evaluación de DQN con el mismo protocolo que experimentos.py.

Uso, desde este directorio:
    python experimentos_dqn.py                 lr, luego gamma, luego epsilon
    python experimentos_dqn.py gamma           sólo una etapa
    python experimentos_dqn.py lr reversion    una etapa y un solo generador

Cada configuración: 3 semillas y el resto de campos de ConfigDQN (dqn_agente.py).
Se elige por log-retorno en validación (50 series, semilla 2024).
El test (100 series, semilla 7777) sólo se guarda para la tabla.
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

from dqn_agente import ConfigDQN, entrenar
from experimentos import N_TEST, N_VAL, N_VAL_CURVA, SEED_TEST, SEED_VAL
from trader_rl import GENERADORES, evaluar, generar_series

DIR = Path(__file__).resolve().parent / "resultados"
SEMILLA_LISTA = [0, 1, 2]
BASE = ConfigDQN()

ETAPAS = {
    "lr": {
        "lr=1e-4": dict(lr=1e-4),
        "lr=3e-4": dict(lr=3e-4),
        "lr=1e-3": dict(lr=1e-3),
        "lr=3e-3": dict(lr=3e-3),
    },
    "gamma": {
        "γ=0.9": dict(gamma=0.9),
        "γ=0.95": dict(gamma=0.95),
        "γ=0.99": dict(gamma=0.99),
    },
    "epsilon": {
        "ε decay (f=0.6)": dict(eps_inicial=1.0, eps_final=0.01, frac_exploracion=0.6),
        "ε decay (f=0.3)": dict(eps_inicial=1.0, eps_final=0.01, frac_exploracion=0.3),
        "ε=0.1 constante": dict(eps_inicial=0.1, eps_final=0.1),
    },
}


def guardar(nombre, datos):
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / f"{nombre}.pkl", "wb") as f:
        pickle.dump(datos, f)


def cargar(nombre):
    with open(DIR / f"{nombre}.pkl", "rb") as f:
        return pickle.load(f)


def entrenar_tarea(tarea):
    """Un entrenamiento. Función de módulo para que el pool pueda importarla."""
    etapa, variante, gen_nombre, cfg, seed = tarea
    torch_ok()
    gen = GENERADORES[gen_nombre]
    val = generar_series(gen, N_VAL, cfg.n_steps, SEED_VAL)
    test = generar_series(gen, N_TEST, cfg.n_steps, SEED_TEST)
    t0 = time.perf_counter()
    etiqueta = f"[{etapa}] {gen_nombre} {variante} semilla {seed}"
    print(f"{etiqueta}  empieza", flush=True)
    agente, hist = entrenar(cfg, gen, seed, series_validacion=val[:N_VAL_CURVA], progreso=etiqueta)
    politica = agente.politica_greedy()
    pesos = {k: v.detach().cpu().clone() for k, v in agente.q.state_dict().items()}
    return {
        "etapa": etapa, "variante": variante, "generador": gen_nombre,
        "algoritmo": "dqn", "cfg": asdict(cfg), "seed": seed,
        "tiempo": time.perf_counter() - t0, "pesos": pesos, "escala": agente.escala,
        "hist": {k: np.asarray(v, dtype=np.float32) for k, v in hist.items()},
        "val": evaluar(politica, val), "test": evaluar(politica, test),
    }


def torch_ok():
    import torch
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def ganadores(resultados):
    """(generador, algoritmo) -> las semillas de la variante con mejor validación."""
    grupos = {}
    for r in resultados:
        grupos.setdefault((r["generador"], r["algoritmo"], r["variante"]), []).append(r)
    mejores = {}
    for (g, a, _), runs in grupos.items():
        puntuacion = np.mean([r["val"]["logret"].mean() for r in runs])
        if (g, a) not in mejores or puntuacion > mejores[(g, a)][0]:
            mejores[(g, a)] = (puntuacion, runs)
    return {k: runs for k, (_, runs) in mejores.items()}


def mejores_configs(resultados):
    return {k: ConfigDQN(**runs[0]["cfg"]) for k, runs in ganadores(resultados).items()}


def ejecutar_etapa(nombre, solo_generador=None):
    orden = list(ETAPAS)
    generadores = (solo_generador,) if solo_generador else tuple(GENERADORES)
    if solo_generador and solo_generador not in GENERADORES:
        raise SystemExit(f"generador desconocido: {solo_generador}. Opciones: {', '.join(GENERADORES)}")
    if nombre == orden[0]:
        base = {g: BASE for g in generadores}
    else:
        previas = mejores_configs(cargar(orden[orden.index(nombre) - 1]))
        base = {g: previas[g, "dqn"] for g in generadores}

    tareas = [(nombre, var, g, replace(base[g], **cambios), s)
              for g in generadores for var, cambios in ETAPAS[nombre].items() for s in SEMILLA_LISTA]
    print(f"[{nombre}] {len(tareas)} entrenamientos, {min(len(tareas), os.cpu_count() or 1)} a la vez. "
          f"Una línea por entrenamiento más o menos cada minuto.", flush=True)
    t0 = time.perf_counter()
    n = min(len(tareas), os.cpu_count() or 1)
    nuevos = []
    with get_context("spawn").Pool(n) as pool:
        for r in pool.imap_unordered(entrenar_tarea, tareas):
            nuevos.append(r)
            val = r["val"]["logret"].mean() * 100
            print(f"[{nombre}] {len(nuevos)}/{len(tareas)} terminados  "
                  f"{r['generador']} {r['variante']} semilla {r['seed']}  "
                  f"{r['tiempo'] / 60:.1f} min  val {val:+.1f}%", flush=True)
    if solo_generador and (DIR / f"{nombre}.pkl").exists():
        previos = [r for r in cargar(nombre) if r["generador"] != solo_generador]
        nuevos = previos + nuevos
    guardar(nombre, nuevos)
    print(f"[{nombre}] terminado en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    for (g, _), cfg in sorted(mejores_configs(nuevos).items()):
        if g not in generadores:
            continue
        eps = f"{cfg.eps_inicial}" if cfg.eps_inicial == cfg.eps_final else f"{cfg.eps_inicial}→{cfg.eps_final}"
        print(f"   {g:10s} lr={cfg.lr:g} γ={cfg.gamma} ε={eps} f={cfg.frac_exploracion}")


if __name__ == "__main__":
    args = sys.argv[1:]
    etapa = args[0] if args else None
    generador = args[1] if len(args) > 1 else None
    pedidas = [etapa] if etapa else list(ETAPAS)
    for nombre in pedidas:
        if nombre not in ETAPAS:
            raise SystemExit(f"etapa desconocida: {nombre}. Opciones: {', '.join(ETAPAS)}")
        ejecutar_etapa(nombre, generador)
