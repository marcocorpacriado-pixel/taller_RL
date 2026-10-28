"""
entrenar.py — Fase 2: entrenamientos nuevos con el código ORIGINAL de Marco (trader_rl.py, sin modificarlo).

Ejecutar con un entorno virtual que tenga gymnasium (ver analisis_adicional/README_REPRODUCCION.md y requirements_entrenamiento.txt),
desde cualquier carpeta y SIN escribir bytecode en el repositorio (-B):

    <python del entorno de entrenamiento> -B -X utf8 analisis_adicional/fase2_evolucion/entrenar.py --grupos REP
    ... --grupos todos            (todos los grupos nuevos; los ya existentes se saltan, nunca se sobrescriben)
    ... --listar                  (muestra el plan y su tamaño sin entrenar)

(Las ejecuciones del 28/10/2026 usaron un entorno en el equipo del autor, en la carpeta de usuario `.venvs/taller_rl_fase2`;
esa ruta es específica de su equipo y no hace falta para reproducir.)

Qué hace:
  * Importa trader_rl.entrenar / Discretizador / evaluar / generar_series tal cual. La única adición es una subclase de
    AgenteTabular que CUENTA actualizaciones por (bin, posición, acción) sin alterar el algoritmo ni consumir números
    aleatorios; se verifica que las corridas duplicadas de Marco (grupo REP) salen idénticas bit a bit.
  * Protocolo idéntico a experimentos.entrenar_y_evaluar: val = 50 series (seed 2024), test = 100 series (seed 7777),
    curva de validación sobre las 10 primeras series cada 100 episodios, semillas de entrenamiento 0..4.
  * Los hiperparámetros NO se eligen aquí: todas las configuraciones están fijadas de antemano en utilidades.plan_entrenamientos().
  * Guarda cada grupo en artefactos/<grupo>.npz (sin pickle). Si el archivo existe no lo toca salvo --forzar.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import os
import platform
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))      # trader_rl y experimentos (código de Marco)
sys.path.insert(0, str(AQUI))      # utilidades

import trader_rl  # noqa: E402
from trader_rl import Config, Discretizador, GENERADORES, entrenar, evaluar, generar_series  # noqa: E402

import utilidades as u  # noqa: E402

_AgenteOriginal = trader_rl.AgenteTabular
SEED_VAL, N_VAL, SEED_TEST, N_TEST, N_VAL_CURVA = u.SEED_VAL, u.N_VAL, u.SEED_TEST, u.N_TEST, 10


class AgenteContador(_AgenteOriginal):
    """AgenteTabular original + contadores de actualizaciones por (bin, posición, acción). No cambia el aprendizaje."""

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.n_total = np.zeros((cfg.n_bins, 2, 3), dtype=np.int64)
        self.n_tarde = np.zeros((cfg.n_bins, 2, 3), dtype=np.int64)   # tras el fin del decaimiento de epsilon
        self._ep = 0
        self._ep_tarde = cfg.frac_exploracion * cfg.n_episodios

    def actualizar(self, s, a, r, s_sig, a_sig, terminal, alpha):
        self.n_total[s[0], s[1], a] += 1
        if self._ep >= self._ep_tarde:
            self.n_tarde[s[0], s[1], a] += 1
        super().actualizar(s, a, r, s_sig, a_sig, terminal, alpha)
        if terminal:
            self._ep += 1


def _iniciar_worker():
    trader_rl.AgenteTabular = AgenteContador


def ejecutar(t):
    """Equivalente a experimentos.entrenar_y_evaluar, con contadores de visitas."""
    cfg = Config(**t["cfg"])
    gen = GENERADORES[t["generador"]]
    disc = Discretizador(gen, cfg.n_bins, cfg.discretizacion)
    val = generar_series(gen, N_VAL, cfg.n_steps, SEED_VAL)
    test = generar_series(gen, N_TEST, cfg.n_steps, SEED_TEST)
    t0 = time.perf_counter()
    agente, hist = entrenar(cfg, gen, disc, t["seed"], series_validacion=val[:N_VAL_CURVA])
    politica = agente.politica_greedy(disc)
    return {
        "id": t["id"], "grupo": t["grupo"], "generador": t["generador"], "algoritmo": t["algoritmo"], "seed": t["seed"],
        "cfg": asdict(cfg), "tiempo": time.perf_counter() - t0, "Q": agente.Q, "cortes": disc.cortes,
        "hist": {k: v.astype(np.float32) for k, v in hist.items()},
        "val": evaluar(politica, val), "test": evaluar(politica, test),
        "visitas_total": agente.n_total, "visitas_tarde": agente.n_tarde,
    }


def comprobar_config_por_defecto():
    """E0 = Config() del código. Comprueba que coincide con DEFAULT (utilidades) y con los valores del enunciado de la Fase 2."""
    real = asdict(Config())
    esperado = {**u.DEFAULT}
    assert real == esperado, f"Config() difiere de DEFAULT: {real} vs {esperado}"
    enunciado = dict(alpha=0.05, gamma=0.95, eps_inicial=1.0, eps_final=0.01, frac_exploracion=0.6, n_episodios=2000, n_bins=10,
                     discretizacion="cuantiles", n_steps=500)
    assert all(real[k] == v for k, v in enunciado.items()), (real, enunciado)
    return real


def comparar_con_marco(resultados, etapas):
    """Compara las corridas del grupo REP con las de Marco (alpha-stage): Q, curvas, val y test deben ser idénticos."""
    por_clave = {(r["generador"], r["algoritmo"], r["variante"], r["seed"]): r for r in etapas["alpha"]}
    filas, max_dif = [], 0.0
    for r in resultados:
        m = por_clave[(r["generador"], r["algoritmo"], r["variante_marco"], r["seed"])]
        d = max(np.abs(r["Q"] - m["Q"]).max(),
                np.abs(r["hist"]["eval_logret"] - m["hist"]["eval_logret"]).max(),
                np.abs(r["hist"]["recompensa"] - m["hist"]["recompensa"]).max(),
                np.abs(r["hist"]["balance"] - m["hist"]["balance"]).max(),
                np.abs(r["val"]["logret"] - m["val"]["logret"]).max(),
                np.abs(r["test"]["logret"] - m["test"]["logret"]).max(),
                np.abs(r["test"]["operaciones"] - m["test"]["operaciones"]).max())
        max_dif = max(max_dif, float(d))
        filas.append(dict(id=r["id"], variante_marco=r["variante_marco"], max_dif_abs=float(d), identica=bool(d == 0)))
    return filas, max_dif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupos", default="REP", help="lista separada por comas, o 'todos'")
    ap.add_argument("--forzar", action="store_true", help="sobrescribe artefactos de Fase 2 existentes (nunca los de Marco)")
    ap.add_argument("--procesos", type=int, default=os.cpu_count())
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    u.asegurar()
    import gymnasium
    cfg0 = comprobar_config_por_defecto()
    etapas = u.marco_etapas()
    plan = u.plan_entrenamientos(etapas)
    if args.listar:
        tot = 0.0
        for g, ts in plan.items():
            eq = sum(t["cfg"]["n_episodios"] for t in ts) / 5000
            tot += eq
            print(f"{g:22s} {len(ts):4d} entrenamientos  (≈{eq:6.1f} equivalentes de 5000 episodios)  {ts[0]['descripcion']}")
        print(f"TOTAL {sum(len(v) for v in plan.values())} entrenamientos ≈ {tot:.1f} equivalentes de 5000 episodios")
        return
    elegidos = list(plan) if args.grupos == "todos" else args.grupos.split(",")
    pendientes = [g for g in elegidos if not (u.ART / f"{g}.npz").exists() or args.forzar]
    for g in elegidos:
        if g not in pendientes:
            print(f"[saltado] {g}: ya existe artefacto (no se sobrescribe sin --forzar)")
    if not pendientes:     # nada que entrenar: no se toca el registro (así verificar_fase2.py V5 no lo modifica)
        return
    u.registrar(f"INICIO python={platform.python_version()} numpy={np.__version__} gymnasium={gymnasium.__version__} "
                f"plataforma={platform.platform()} procesos={args.procesos} grupos={pendientes}")

    # REP primero: es la puerta de reproducibilidad. Si no es idéntica a Marco se aborta antes de entrenar el resto.
    orden = (["REP"] if "REP" in pendientes else []) + sorted([g for g in pendientes if g != "REP"],
                                                                key=lambda g: -plan[g][0]["cfg"]["n_episodios"])
    for g in orden:
        ts = plan[g]
        t0 = time.perf_counter()
        print(f"[{g}] {len(ts)} entrenamientos (N={ts[0]['cfg']['n_episodios']})...", flush=True)
        u.registrar(f"GRUPO {g} inicio: {len(ts)} entrenamientos, {ts[0]['descripcion']}")
        with Pool(min(args.procesos, len(ts)), initializer=_iniciar_worker) as pool:
            res = pool.map(ejecutar, ts, chunksize=1)
        if g == "REP":
            filas, max_dif = comparar_con_marco([dict(r, variante_marco=t["variante_marco"]) for r, t in zip(res, ts)], etapas)
            import csv
            with open(u.RES / "verificacion_reproduccion_marco.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(filas[0]))
                w.writeheader()
                w.writerows(filas)
            n_ident = sum(f["identica"] for f in filas)
            u.registrar(f"REP: {n_ident}/{len(filas)} corridas idénticas a Marco (max|dif|={max_dif})")
            print(f"[REP] idénticas a Marco: {n_ident}/{len(filas)}  max|dif|={max_dif}", flush=True)
            if max_dif != 0:
                print("ABORTO: el entorno actual NO reproduce exactamente los artefactos de Marco; no se entrena el resto.", flush=True)
                u.registrar("ABORTO: REP no reproduce a Marco")
                sys.exit(2)
        ruta = u.guardar_grupo(g, res, forzar=args.forzar)
        dt = time.perf_counter() - t0
        u.registrar(f"GRUPO {g} fin: {dt / 60:.1f} min -> {ruta.name}")
        print(f"[{g}] terminado en {dt / 60:.1f} min -> {ruta.name}", flush=True)


if __name__ == "__main__":
    main()
