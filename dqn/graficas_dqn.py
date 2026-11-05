"""
Figuras y tablas de DQN, con las mismas columnas que figuras/tabla_resultados.md.

Uso, desde este directorio, cuando ya existe resultados/epsilon.pkl:
    python graficas_dqn.py
"""
import sys
from pathlib import Path

import matplotlib

try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from dqn_agente import ConfigDQN, cargar_pesos
from estilo import COLOR, NOMBRE, TINTA, TITULO, aplicar_estilo
from experimentos_dqn import DIR, ETAPAS, SEMILLA_LISTA, cargar, ganadores
from trader_rl import (GENERADORES, evaluar, generar_series, politica_aleatoria,
                       politica_buy_and_hold)

DIR_FIG = Path(__file__).resolve().parent / "figuras"
DIR_TABULAR = RAIZ / "resultados"
AGENTES_TABLA = ("aleatorio", "buy&hold", "qlearning", "sarsa", "dqn")


def _media_movil(x, n):
    return np.convolve(x, np.ones(n) / n, mode="valid")


def episodios_convergencia(episodios, curva):
    """Misma definición que graficas.py: 90 % del recorrido hasta la meseta final."""
    suave = _media_movil(curva, 3)
    meseta = suave[-max(1, len(suave) // 5):].mean()
    umbral = suave[0] + 0.9 * (meseta - suave[0])
    alcanzado = suave >= umbral if meseta >= suave[0] else suave <= umbral
    return int(episodios[1:-1][np.argmax(alcanzado)])


def _guardar(fig, nombre):
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIG / nombre)
    plt.close(fig)
    print("  ", nombre)


def _cargar_tabular(nombre):
    import pickle
    with open(DIR_TABULAR / f"{nombre}.pkl", "rb") as f:
        return pickle.load(f)


def fig_ajuste(etapas):
    fig, axs = plt.subplots(len(etapas), 4, figsize=(15, 3 * len(etapas)), squeeze=False)
    for i, (etapa, resultados) in enumerate(etapas.items()):
        variantes = list(ETAPAS[etapa])
        elegido = {g: runs[0]["variante"] for (g, _), runs in ganadores(resultados).items()}
        for j, g in enumerate(GENERADORES):
            ax = axs[i, j]
            runs_v = [[r for r in resultados if r["generador"] == g and r["variante"] == v] for v in variantes]
            x = np.arange(len(variantes))
            medias = [np.mean([r["val"]["logret"].mean() * 100 for r in rs]) for rs in runs_v]
            desv = [np.std([r["val"]["logret"].mean() * 100 for r in rs]) for rs in runs_v]
            ax.errorbar(x, medias, yerr=desv, fmt="o", ms=6, capsize=3, color=COLOR["dqn"])
            e = variantes.index(elegido[g])
            ax.plot(x[e], medias[e], "o", ms=13, mfc="none", mec=COLOR["dqn"], mew=1.5)
            ax.set_xticks(x, variantes, fontsize=8)
            if i == 0:
                ax.set_title(TITULO[g])
            if j == 0:
                ax.set_ylabel(f"etapa {etapa}\nlog-ret. validación (%)")
    fig.suptitle(f"DQN: ajuste secuencial (media ± desv. entre {len(SEMILLA_LISTA)} semillas; círculo = elegido)", color=TINTA)
    fig.tight_layout()
    _guardar(fig, "fig_ajuste_dqn.png")


def fig_convergencia(finales):
    from experimentos import N_VAL, N_VAL_CURVA, SEED_VAL
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, g in zip(axs, GENERADORES):
        runs = finales[(g, "dqn")]
        ep = runs[0]["hist"]["eval_episodio"]
        curvas = np.array([r["hist"]["eval_logret"] for r in runs]) * 100
        m, s = curvas.mean(0), curvas.std(0)
        ax.plot(ep, m, color=COLOR["dqn"], label=NOMBRE["dqn"])
        ax.fill_between(ep, m - s, m + s, color=COLOR["dqn"], alpha=0.15, lw=0)
        serie = generar_series(GENERADORES[g], N_VAL, 500, SEED_VAL)[:N_VAL_CURVA]
        for b, pol in (("buy&hold", politica_buy_and_hold), ("aleatorio", politica_aleatoria(0))):
            ax.axhline(evaluar(pol, serie)["logret"].mean() * 100, ls="--", lw=1.2, color=COLOR[b], label=NOMBRE[b])
        ax.set_title(TITULO[g])
        ax.set_xlabel("episodio de entrenamiento")
    axs[0].set_ylabel("log-ret. validación greedy (%)")
    axs[0].legend(loc="best")
    fig.suptitle(f"DQN: política greedy cada 100 episodios (media ± desv. entre {len(SEMILLA_LISTA)} semillas)", color=TINTA, y=1.03)
    fig.tight_layout()
    _guardar(fig, "fig_convergencia_dqn.png")


def fig_test(finales, baselines):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, g in zip(axs, GENERADORES):
        datos = [np.concatenate([r["logret"] for r in baselines[(g, b)]]) for b in ("aleatorio", "buy&hold")]
        datos.append(np.concatenate([r["test"]["logret"] for r in finales[(g, "dqn")]]))
        colores = ["aleatorio", "buy&hold", "dqn"]
        cajas = ax.boxplot([d * 100 for d in datos], patch_artist=True, widths=0.6, showfliers=False,
                           medianprops={"color": "#0b0b0b", "lw": 1.5})
        for caja, nombre in zip(cajas["boxes"], colores):
            caja.set(facecolor=COLOR[nombre], alpha=0.75, edgecolor=COLOR[nombre])
        ax.plot(range(1, 4), [d.mean() * 100 for d in datos], "D", color="#0b0b0b", ms=5)
        ax.axhline(0, color="#52514e", lw=1)
        ax.set_xticks(range(1, 4), [NOMBRE[a] for a in colores], fontsize=8, rotation=15, ha="right")
        ax.set_title(TITULO[g])
    axs[0].set_ylabel("log-retorno en test (%)")
    fig.suptitle(f"DQN en test (100 series × {len(SEMILLA_LISTA)} semillas; cajas sin outliers)", color="#0b0b0b", y=1.03)
    fig.tight_layout()
    _guardar(fig, "fig_test_dqn.png")


def fig_senal(finales):
    """Una figura por el mismo criterio que fig_senal_politica: fracción de semillas que opera."""
    from visualizar import _centros_por_bin, _senal
    from dqn_agente import COMPRAR, MANTENER, VENDER
    from trader_rl import Discretizador
    from estilo import POSITIVO, NEGATIVO, TINTA, TINTA_2

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.7], hspace=0.08, wspace=0.28)
    filas = ((0, "fuera: compra"), (1, "dentro: vende"))
    for j, g in enumerate(GENERADORES):
        media, se = _senal(g)
        n_bins = len(media)
        cortes = Discretizador(GENERADORES[g], n_bins).cortes
        centros = _centros_por_bin(g, cortes)
        ax = fig.add_subplot(gs[0, j])
        ax.bar(range(n_bins), media * 100, yerr=2 * se * 100, width=0.75,
               color=[POSITIVO if m > 0 else NEGATIVO for m in media],
               error_kw={"lw": 1, "ecolor": TINTA_2})
        for y in (0.2, -0.2):
            ax.axhline(y, ls="--", lw=1, color=TINTA_2)
        ax.set_title(TITULO[g])
        ax.set_xticks([])
        if j == 0:
            ax.set_ylabel("log-retorno 10 pasos (%)")
        mapa = np.zeros((2, n_bins))
        runs = finales[(g, "dqn")]
        for run in runs:
            agente = cargar_pesos(ConfigDQN(**run["cfg"]), run["seed"], run["pesos"], run["escala"])
            for i, (pos, _) in enumerate(filas):
                for b, tendencia in enumerate(centros):
                    q = agente.valores(np.array([tendencia, pos], np.float32))
                    accion, alternativa = (COMPRAR, MANTENER) if pos == 0 else (VENDER, MANTENER)
                    d = q[accion] - q[alternativa]
                    mapa[i, b] += (d > 0) + 0.5 * (d == 0)
        mapa /= len(runs)
        axp = fig.add_subplot(gs[1, j])
        axp.imshow(mapa, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, b), v in np.ndenumerate(mapa):
            if v >= 0.5:
                axp.text(b, i, "C" if i == 0 else "V", ha="center", va="center", fontsize=7,
                         color="white" if v > 0.7 else TINTA)
        axp.set_yticks(range(2), [nombre for _, nombre in filas] if j == 0 else [])
        axp.set_xticks(range(n_bins))
        axp.grid(False)
    fig.supxlabel("bin de tendencia (0 = caída fuerte, 9 = subida fuerte)", fontsize=9, color=TINTA_2)
    fig.suptitle(f"Señal disponible frente a la política DQN (fracción de {len(SEMILLA_LISTA)} semillas)", color=TINTA, y=0.99)
    _guardar(fig, "fig_senal_politica_dqn.png")


def _fila_metricas(runs, clave_test=True):
    if clave_test:
        lr = np.array([r["test"]["logret"].mean() * 100 for r in runs])
        bal = np.concatenate([r["test"]["balance_final"] for r in runs])
        ops = np.concatenate([r["test"]["operaciones"] for r in runs])
    else:
        lr = np.array([r["logret"].mean() * 100 for r in runs])
        bal = np.concatenate([r["balance_final"] for r in runs])
        ops = np.concatenate([r["operaciones"] for r in runs])
    return lr, bal, ops


def _linea(generador, nombre, lr, bal, ops, extra):
    return (f"| {TITULO[generador]} | {nombre} | {lr.mean():+.1f} ± {lr.std():.1f} | "
            f"{np.median(bal):,.0f} | {ops.mean():.0f} | {extra} |")


def tabla(finales, baselines):
    filas = ["| Generador | Agente | Log-ret. test (%) | Mediana balance ($) | Operaciones | "
             "Episodios hasta convergencia | Tiempo (s) | lr · γ · ε |",
             "|---|---|---|---|---|---|---|---|"]
    for g in GENERADORES:
        for b in ("aleatorio", "buy&hold"):
            lr, bal, ops = _fila_metricas(baselines[(g, b)], clave_test=False)
            filas.append(_linea(g, NOMBRE[b], lr, bal, ops, "— | — | —"))
        runs = finales[(g, "dqn")]
        lr, bal, ops = _fila_metricas(runs)
        curva = np.mean([r["hist"]["eval_logret"] for r in runs], axis=0)
        conv = episodios_convergencia(runs[0]["hist"]["eval_episodio"], curva)
        c = ConfigDQN(**runs[0]["cfg"])
        eps = f"{c.eps_inicial}" if c.eps_inicial == c.eps_final else f"1→0.01 (f={c.frac_exploracion})"
        tiempo = np.mean([r["tiempo"] for r in runs])
        extra = f"{conv} | {tiempo:.0f} | {c.lr:g} · {c.gamma} · {eps}"
        filas.append(_linea(g, f"**{NOMBRE['dqn']}**", lr, bal, ops, extra))
    texto = "\n".join(filas)
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    (DIR_FIG / "tabla_resultados_dqn.md").write_text(
        texto + "\n\nLog-ret.: media ± desv. entre semillas de la media sobre las series de test.\n",
        encoding="utf-8")
    print(texto)
    return texto


def tabla_comparacion():
    """Junta DQN con la tabla de Q-Learning y SARSA. Misma métrica y mismas series."""
    if not (DIR / "epsilon.pkl").exists():
        print("Todavía no hay resultados de DQN. Desde el directorio dqn:\n"
              "    python experimentos_dqn.py\n"
              "    python graficas_dqn.py")
        return None
    if not (DIR_TABULAR / "epsilon.pkl").exists() or not (DIR_TABULAR / "baselines.pkl").exists():
        print("No están los resultados tabulares en resultados/.")
        return None
    from experimentos import ganadores as ganadores_tab
    finales_tab = ganadores_tab(_cargar_tabular("epsilon"))
    baselines = _cargar_tabular("baselines")
    finales = ganadores(cargar("epsilon"))
    filas = ["| Generador | Agente | Log-ret. test (%) | Mediana balance ($) | Operaciones | "
             "Episodios hasta convergencia | Tiempo (s) |",
             "|---|---|---|---|---|---|---|"]
    for g in GENERADORES:
        for b in ("aleatorio", "buy&hold"):
            lr, bal, ops = _fila_metricas(baselines[(g, b)], clave_test=False)
            filas.append(f"| {TITULO[g]} | {NOMBRE[b]} | {lr.mean():+.1f} ± {lr.std():.1f} | "
                         f"{np.median(bal):,.0f} | {ops.mean():.0f} | — | — |")
        for algo in ("qlearning", "sarsa"):
            runs = finales_tab[(g, algo)]
            lr = np.array([r["test"]["logret"].mean() * 100 for r in runs])
            bal = np.concatenate([r["test"]["balance_final"] for r in runs])
            ops = np.concatenate([r["test"]["operaciones"] for r in runs])
            curva = np.mean([r["hist"]["eval_logret"] for r in runs], axis=0)
            conv = episodios_convergencia(runs[0]["hist"]["eval_episodio"], curva)
            tiempo = np.mean([r["tiempo"] for r in runs])
            filas.append(f"| {TITULO[g]} | {NOMBRE[algo]} | {lr.mean():+.1f} ± {lr.std():.1f} | "
                         f"{np.median(bal):,.0f} | {ops.mean():.0f} | {conv} | {tiempo:.0f} |")
        runs = finales[(g, "dqn")]
        lr, bal, ops = _fila_metricas(runs)
        curva = np.mean([r["hist"]["eval_logret"] for r in runs], axis=0)
        conv = episodios_convergencia(runs[0]["hist"]["eval_episodio"], curva)
        tiempo = np.mean([r["tiempo"] for r in runs])
        filas.append(f"| {TITULO[g]} | **{NOMBRE['dqn']}** | {lr.mean():+.1f} ± {lr.std():.1f} | "
                     f"{np.median(bal):,.0f} | {ops.mean():.0f} | {conv} | {tiempo:.0f} |")
    texto = "\n".join(filas)
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    (DIR_FIG / "tabla_comparacion.md").write_text(
        texto + "\n\nMisma métrica, mismas series de test (semilla 7777, 100 series) y mismos baselines.\n",
        encoding="utf-8")
    print(texto)
    return texto


def main():
    aplicar_estilo()
    etapas = {e: cargar(e) for e in ETAPAS}
    finales = ganadores(etapas["epsilon"])
    baselines = _cargar_tabular("baselines")
    fig_ajuste(etapas)
    fig_convergencia(finales)
    fig_test(finales, baselines)
    fig_senal(finales)
    tabla(finales, baselines)
    print()
    tabla_comparacion()


if __name__ == "__main__":
    main()
