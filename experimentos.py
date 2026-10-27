"""
experimentos.py — Ajuste de hiperparámetros, evaluación final y análisis de señal.

Uso:
    py experimentos.py              ejecuta todas las etapas en orden
    py experimentos.py gamma        ejecuta sólo una etapa (lee las anteriores de resultados/)

Protocolo (BITACORA D13):
  * Ajuste SECUENCIAL por generador y algoritmo: α -> γ -> ε. Cada etapa parte de
    la mejor configuración de la anterior (un grid completo sería ~5x más caro).
  * Cada configuración se entrena con 5 semillas (media ± desviación, D4).
  * Selección por log-retorno medio en VALIDACIÓN (50 series). El TEST (100 series)
    sólo se reporta: elegir con test inflaría los resultados (data leakage).
  * La configuración final de cada (generador, algoritmo) es la ganadora de la
    última etapa; sus 5 semillas son los resultados que se reportan.

Salida: resultados/<etapa>.pkl (lista de dicts, uno por entrenamiento).
"""
import os
import pickle
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

from trader_rl import (GENERADORES, Config, Discretizador, calcular_tendencias, entrenar,
                       evaluar, generar_series, politica_aleatoria, politica_buy_and_hold)

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
ALGORITMOS = ("qlearning", "sarsa")
SEMILLAS = range(5)
SEED_VAL, N_VAL = 2024, 50       # validación: selección de hiperparámetros
SEED_TEST, N_TEST = 7777, 100    # test: sólo para reportar
N_VAL_CURVA = 10                 # subconjunto de validación para la curva de convergencia
BASE = Config(n_episodios=5000)  # punto de partida (D12)

# Variantes de cada etapa: nombre -> cambios sobre la mejor configuración previa.
ETAPAS = {
    "alpha": {
        "α=0.001": dict(alpha=0.001),
        "α=0.002": dict(alpha=0.002),
        "α=0.005": dict(alpha=0.005),
        "α=0.01": dict(alpha=0.01),
        "α=0.05→0.001": dict(alpha=0.05, alpha_decay=0.999, alpha_min=0.001),
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

# Experimentos de análisis: parten de la configuración final (última etapa de
# ETAPAS) pero NO la modifican; sirven para justificar decisiones fijas.
EXTRAS = {
    "discretizacion": {
        "5 cuantiles": dict(n_bins=5),
        "10 cuantiles": dict(n_bins=10),
        "20 cuantiles": dict(n_bins=20),
        "10 uniformes": dict(n_bins=10, discretizacion="uniforme"),
    },
}
VARIANTES = {**ETAPAS, **EXTRAS}


def guardar(nombre, datos):
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, f"{nombre}.pkl"), "wb") as f:
        pickle.dump(datos, f)


def cargar(nombre):
    with open(os.path.join(DIR, f"{nombre}.pkl"), "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Un entrenamiento completo (se ejecuta en paralelo, un proceso por tarea)
# -----------------------------------------------------------------------------
def entrenar_y_evaluar(tarea):
    etapa, variante, gen_nombre, cfg, seed = tarea
    gen = GENERADORES[gen_nombre]
    disc = Discretizador(gen, cfg.n_bins, cfg.discretizacion)
    val = generar_series(gen, N_VAL, cfg.n_steps, SEED_VAL)
    test = generar_series(gen, N_TEST, cfg.n_steps, SEED_TEST)

    t0 = time.perf_counter()
    agente, hist = entrenar(cfg, gen, disc, seed, series_validacion=val[:N_VAL_CURVA])
    politica = agente.politica_greedy(disc)
    return {
        "etapa": etapa, "variante": variante, "generador": gen_nombre,
        "algoritmo": cfg.algoritmo, "cfg": asdict(cfg), "seed": seed,
        "tiempo": time.perf_counter() - t0, "Q": agente.Q, "cortes": disc.cortes,
        "hist": {k: v.astype(np.float32) for k, v in hist.items()},
        "val": evaluar(politica, val), "test": evaluar(politica, test),
    }


def ganadores(resultados):
    """(generador, algoritmo) -> entrenamientos (las 5 semillas) de la variante con
    mayor log-retorno medio en VALIDACIÓN."""
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
    """(generador, algoritmo) -> Config ganadora."""
    return {k: Config(**runs[0]["cfg"]) for k, runs in ganadores(resultados).items()}


def ejecutar_etapa(nombre):
    """Entrena todas las variantes de una etapa partiendo de la mejor config previa
    (para un experimento de EXTRAS, de la configuración final)."""
    orden = list(ETAPAS)
    if nombre == orden[0]:
        base = {(g, a): replace(BASE, algoritmo=a) for g in GENERADORES for a in ALGORITMOS}
    elif nombre in ETAPAS:
        base = mejores_configs(cargar(orden[orden.index(nombre) - 1]))
    else:
        base = mejores_configs(cargar(orden[-1]))

    tareas = [(nombre, var, g, replace(base[(g, a)], **cambios), s)
              for (g, a) in base for var, cambios in VARIANTES[nombre].items() for s in SEMILLAS]
    print(f"[{nombre}] {len(tareas)} entrenamientos...", flush=True)
    t0 = time.perf_counter()
    with Pool(min(len(tareas), os.cpu_count())) as pool:
        resultados = pool.map(entrenar_y_evaluar, tareas)
    guardar(nombre, resultados)
    print(f"[{nombre}] terminado en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    for (g, a), cfg in sorted(mejores_configs(resultados).items()):
        print(f"   {g:10s} {a:9s} -> α={cfg.alpha} decay={cfg.alpha_decay} γ={cfg.gamma} "
              f"ε={cfg.eps_inicial}→{cfg.eps_final} f={cfg.frac_exploracion}")


# -----------------------------------------------------------------------------
# Baselines y análisis de la señal disponible
# -----------------------------------------------------------------------------
def ejecutar_baselines():
    """Agente aleatorio (5 semillas) y Buy & Hold sobre las mismas series de test."""
    res = {}
    for g, gen in GENERADORES.items():
        test = generar_series(gen, N_TEST, BASE.n_steps, SEED_TEST)
        res[(g, "aleatorio")] = [evaluar(politica_aleatoria(s), test) for s in SEMILLAS]
        res[(g, "buy&hold")] = [evaluar(politica_buy_and_hold, test)]
    guardar("baselines", res)
    print("[baselines] terminado", flush=True)


def senal_por_bin(gen, n_bins=10, k=10, n_series=400, seed=55):
    """Log-retorno medio de los k pasos siguientes, condicionado al bin de tendencia (H3).

    Es la "verdad de referencia": indica qué política es explotable con este estado.
    Devuelve (media, error estándar) por bin.
    """
    disc = Discretizador(gen, n_bins)
    por_bin = [[] for _ in range(n_bins)]
    for p in generar_series(gen, n_series, BASE.n_steps, seed):
        n = len(p) - 5 - k                               # pasos con k precios futuros
        bins = np.digitize(calcular_tendencias(p)[:n], disc.cortes)
        futuro = np.log(p[5 + k: 5 + k + n] / p[5: 5 + n])
        for b in range(n_bins):
            por_bin[b].extend(futuro[bins == b])
    media = np.array([np.mean(x) for x in por_bin])
    se = np.array([np.std(x) / np.sqrt(len(x)) for x in por_bin])
    return media, se


def ejecutar_senal():
    guardar("senal", {g: senal_por_bin(gen) for g, gen in GENERADORES.items()})
    print("[senal] terminado", flush=True)


if __name__ == "__main__":
    pedidas = sys.argv[1:] or ["senal", "baselines", *VARIANTES]
    for etapa in pedidas:
        {"senal": ejecutar_senal, "baselines": ejecutar_baselines}.get(etapa, lambda: ejecutar_etapa(etapa))()
