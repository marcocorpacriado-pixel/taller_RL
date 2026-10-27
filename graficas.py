"""
graficas.py — Figuras y tabla de resultados a partir de resultados/*.pkl.

Uso:  py graficas.py      (genera figuras/*.png y figuras/tabla_resultados.md)

Cada figura responde a un punto de la rúbrica:
  fig_generadores.png         contexto: una serie de ejemplo de cada generador
  fig_senal_politica.png      análisis crítico: señal disponible vs. política aprendida
  fig_ajuste.png              justificación de α, γ y ε (validación, 5 semillas)
  fig_discretizacion.png      justificación del nº de bins y de los cuantiles
  fig_convergencia.png        convergencia Q-Learning vs SARSA (validación greedy)
  fig_convergencia_eps.png    ε con decaimiento vs. constante
  fig_pnl_entrenamiento.png   PnL por episodio durante el entrenamiento
  fig_test.png                distribución del log-retorno en test vs. baselines
  fig_senales.png             señales de compra/venta y balance sobre una serie de test
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experimentos import (ALGORITMOS, ETAPAS, N_VAL_CURVA, SEED_TEST, SEED_VAL, N_TEST, N_VAL,
                          VARIANTES, cargar, ganadores)
from trader_rl import (COMPRAR, GENERADORES, MANTENER, VENDER, AgenteTabular, Config, Discretizador,
                       ejecutar_episodio, evaluar, generar_series, politica_aleatoria,
                       politica_buy_and_hold)

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")

# Un color fijo por agente en TODAS las figuras (paleta categórica validada para daltonismo).
COLOR = {"qlearning": "#2a78d6", "sarsa": "#eb6834", "buy&hold": "#1baf7a", "aleatorio": "#8a8984"}
NOMBRE = {"qlearning": "Q-Learning", "sarsa": "SARSA", "buy&hold": "Buy & Hold", "aleatorio": "Aleatorio"}
TITULO = {"ciclico": "Tendencia + ciclo", "browniano": "Browniano geométrico",
          "reversion": "Reversión a la media", "regimenes": "Regímenes alcista/bajista"}
TINTA, TINTA_2, POSITIVO, NEGATIVO = "#0b0b0b", "#52514e", "#2a78d6", "#e34948"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
})


def guardar(fig, nombre):
    os.makedirs(DIR, exist_ok=True)
    fig.savefig(os.path.join(DIR, nombre))
    plt.close(fig)
    print("  ", nombre)


def media_movil(x, n):
    return np.convolve(x, np.ones(n) / n, mode="valid")


def series_curva(g):
    """Las mismas series de validación que usó la curva de convergencia durante el entrenamiento."""
    return generar_series(GENERADORES[g], N_VAL, 500, SEED_VAL)[:N_VAL_CURVA]


def episodios_convergencia(episodios, curva):
    """Primer episodio en el que la curva (media móvil de 3 puntos) cubre el 90 % del
    recorrido entre su valor inicial y su meseta final (media del último 20 %).

    ponytail: criterio heurístico sobre una curva ruidosa; un test estadístico de
    estacionariedad sería más riguroso si hiciera falta precisión.
    """
    suave = media_movil(curva, 3)
    meseta = suave[-max(1, len(suave) // 5):].mean()
    umbral = suave[0] + 0.9 * (meseta - suave[0])
    alcanzado = suave >= umbral if meseta >= suave[0] else suave <= umbral
    return int(episodios[1:-1][np.argmax(alcanzado)])


def politica_de(run):
    """Reconstruye la política greedy de un entrenamiento guardado (Q + cortes)."""
    cfg = Config(**run["cfg"])
    agente = AgenteTabular(cfg, np.random.default_rng(0))
    agente.Q = run["Q"]
    disc = Discretizador(GENERADORES[run["generador"]], cfg.n_bins, cfg.discretizacion)
    assert np.allclose(disc.cortes, run["cortes"])
    return agente.politica_greedy(disc)


# -----------------------------------------------------------------------------
def fig_generadores():
    fig, axs = plt.subplots(1, 4, figsize=(14, 2.6))
    for ax, (g, gen) in zip(axs, GENERADORES.items()):
        for i, p in enumerate(generar_series(gen, 3, 500, seed=SEED_TEST)):
            ax.plot(p, lw=1.2, color=TINTA if i == 0 else "#b5b4ae", zorder=3 - i)
        ax.set_title(TITULO[g])
        ax.set_xlabel("paso")
        if g == "ciclico":
            ax.set_yscale("log")
    axs[0].set_ylabel("precio")
    guardar(fig, "fig_generadores.png")


def fig_senal_politica(finales):
    """Arriba: lo que SE PUEDE aprender (log-retorno futuro por bin).
    Abajo: lo que SE HA aprendido (fracción de semillas que compra / vende en cada bin)."""
    senal = cargar("senal")
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.7], hspace=0.08, wspace=0.28)
    filas = [(a, pos) for a in ALGORITMOS for pos in (0, 1)]
    for j, g in enumerate(GENERADORES):
        media, se = senal[g]
        n_bins = len(media)
        ax = fig.add_subplot(gs[0, j])
        ax.bar(range(n_bins), media * 100, yerr=2 * se * 100, width=0.75, error_kw={"lw": 1, "ecolor": TINTA_2},
               color=[POSITIVO if m > 0 else NEGATIVO for m in media])
        for y in (0.2, -0.2):
            ax.axhline(y, ls="--", lw=1, color=TINTA_2)
        ax.text(n_bins - 0.5, 0.2, "coste ida+vuelta", ha="right", va="bottom", fontsize=7, color=TINTA_2)
        ax.set_title(TITULO[g])
        ax.set_xticks([])
        if j == 0:
            ax.set_ylabel("log-retorno 10 pasos (%)")

        # Fracción de las 5 semillas que elige operar en cada (bin, posición). Empate = 0.5.
        mapa = np.zeros((len(filas), n_bins))
        for i, (a, pos) in enumerate(filas):
            for run in finales[(g, a)]:
                Q = run["Q"]
                accion, alternativa = (COMPRAR, MANTENER) if pos == 0 else (VENDER, MANTENER)
                d = Q[:, pos, accion] - Q[:, pos, alternativa]
                mapa[i] += (d > 0) + 0.5 * (d == 0)
            mapa[i] /= len(finales[(g, a)])
        axp = fig.add_subplot(gs[1, j])
        axp.imshow(mapa, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, b), v in np.ndenumerate(mapa):
            if v >= 0.5:
                axp.text(b, i, "C" if filas[i][1] == 0 else "V", ha="center", va="center",
                         fontsize=7, color="white" if v > 0.7 else TINTA)
        axp.set_yticks(range(len(filas)))
        axp.set_yticklabels([f"{NOMBRE[a]} · {'fuera: compra' if p == 0 else 'dentro: vende'}" for a, p in filas]
                            if j == 0 else [])
        axp.set_xticks(range(n_bins))
        axp.grid(False)
    fig.supxlabel("bin de tendencia (0 = caída fuerte, 9 = subida fuerte)", fontsize=9, color=TINTA_2, y=0.0)
    fig.suptitle("Señal disponible (arriba, ±2 e.s.) frente a política aprendida (abajo, fracción de 5 semillas)",
                 y=0.99, fontsize=10, color=TINTA)
    guardar(fig, "fig_senal_politica.png")


def fig_ajuste(etapas, nombre="fig_ajuste.png",
               titulo="Ajuste secuencial de hiperparámetros (media ± desv. entre 5 semillas; círculo = elegido)"):
    fig, axs = plt.subplots(len(etapas), 4, figsize=(15, 3 * len(etapas)), squeeze=False)
    for i, (etapa, resultados) in enumerate(etapas.items()):
        variantes = list(VARIANTES[etapa])
        elegido = {k: runs[0]["variante"] for k, runs in ganadores(resultados).items()}
        for j, g in enumerate(GENERADORES):
            ax = axs[i, j]
            for k, a in enumerate(ALGORITMOS):
                x = np.arange(len(variantes)) + (k - 0.5) * 0.22
                por_semilla = [[r["val"]["logret"].mean() * 100 for r in resultados
                                if r["generador"] == g and r["algoritmo"] == a and r["variante"] == v]
                               for v in variantes]
                ax.errorbar(x, [np.mean(s) for s in por_semilla], yerr=[np.std(s) for s in por_semilla],
                            fmt="o", ms=6, capsize=3, color=COLOR[a], label=NOMBRE[a])
                e = variantes.index(elegido[(g, a)])
                ax.plot(x[e], np.mean(por_semilla[e]), "o", ms=13, mfc="none", mec=COLOR[a], mew=1.5)
            ax.set_xticks(range(len(variantes)))
            ax.set_xticklabels(variantes, fontsize=8)
            if i == 0:
                ax.set_title(TITULO[g])
            if j == 0:
                ax.set_ylabel(f"etapa {etapa}\nlog-ret. validación (%)")
    axs[0, 0].legend(loc="best")
    fig.suptitle(titulo, y=1.0, fontsize=10, color=TINTA)
    fig.tight_layout()
    guardar(fig, nombre)


def fig_convergencia(finales, baselines_curva):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, g in zip(axs, GENERADORES):
        for a in ALGORITMOS:
            runs = finales[(g, a)]
            ep = runs[0]["hist"]["eval_episodio"]
            curvas = np.array([r["hist"]["eval_logret"] for r in runs]) * 100
            m, s = curvas.mean(0), curvas.std(0)
            ax.plot(ep, m, color=COLOR[a], label=NOMBRE[a])
            ax.fill_between(ep, m - s, m + s, color=COLOR[a], alpha=0.15, lw=0)
        for b in ("buy&hold", "aleatorio"):
            ax.axhline(baselines_curva[(g, b)] * 100, ls="--", lw=1.2, color=COLOR[b], label=NOMBRE[b])
        ax.set_title(TITULO[g])
        ax.set_xlabel("episodio de entrenamiento")
    axs[0].set_ylabel("log-ret. validación greedy (%)")
    axs[0].legend(loc="lower right")
    fig.suptitle("Convergencia: política greedy evaluada cada 100 episodios (media ± desv. entre 5 semillas)",
                 y=1.03, fontsize=10, color=TINTA)
    guardar(fig, "fig_convergencia.png")


def fig_convergencia_eps(resultados):
    colores = ["#008300", "#6250d6", "#eda100"]
    fig, axs = plt.subplots(2, 4, figsize=(15, 5.6), sharex=True)
    for i, a in enumerate(ALGORITMOS):
        for j, g in enumerate(GENERADORES):
            ax = axs[i, j]
            for c, v in zip(colores, ETAPAS["epsilon"]):
                runs = [r for r in resultados if r["generador"] == g and r["algoritmo"] == a and r["variante"] == v]
                curvas = np.array([r["hist"]["eval_logret"] for r in runs]) * 100
                ax.plot(runs[0]["hist"]["eval_episodio"], curvas.mean(0), color=c, label=v)
            if i == 0:
                ax.set_title(TITULO[g])
            if j == 0:
                ax.set_ylabel(f"{NOMBRE[a]}\nlog-ret. validación (%)")
            if i == 1:
                ax.set_xlabel("episodio de entrenamiento")
    axs[0, 0].legend(loc="lower right")
    fig.suptitle("Exploración: ε con decaimiento (f = fracción de episodios hasta ε=0.01) frente a ε constante",
                 y=1.0, fontsize=10, color=TINTA)
    fig.tight_layout()
    guardar(fig, "fig_convergencia_eps.png")


def fig_pnl_entrenamiento(finales):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, g in zip(axs, GENERADORES):
        for a in ALGORITMOS:
            pnl = np.array([np.log(r["hist"]["balance"] / 1000.0) for r in finales[(g, a)]]) * 100
            ax.plot(np.arange(100, pnl.shape[1] + 1), media_movil(pnl.mean(0), 100), color=COLOR[a], label=NOMBRE[a])
        ax.axhline(0, color=TINTA_2, lw=1)
        ax.set_title(TITULO[g])
        ax.set_xlabel("episodio")
    axs[0].set_ylabel("log-retorno del episodio (%)\nmedia móvil 100, con exploración")
    axs[0].legend(loc="lower right")
    guardar(fig, "fig_pnl_entrenamiento.png")


def fig_test(finales, baselines):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    agentes = ["aleatorio", "buy&hold", *ALGORITMOS]
    for ax, g in zip(axs, GENERADORES):
        datos = [np.concatenate([r["logret"] for r in baselines[(g, b)]]) for b in agentes[:2]]
        datos += [np.concatenate([r["test"]["logret"] for r in finales[(g, a)]]) for a in ALGORITMOS]
        cajas = ax.boxplot([d * 100 for d in datos], patch_artist=True, widths=0.6, showfliers=False,
                           medianprops={"color": TINTA, "lw": 1.5})
        for caja, a in zip(cajas["boxes"], agentes):
            caja.set(facecolor=COLOR[a], alpha=0.75, edgecolor=COLOR[a])
        ax.plot(range(1, 5), [d.mean() * 100 for d in datos], "D", color=TINTA, ms=5, label="media")
        ax.axhline(0, color=TINTA_2, lw=1)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([NOMBRE[a] for a in agentes], fontsize=8, rotation=20, ha="right")
        ax.set_title(TITULO[g])
    axs[0].set_ylabel("log-retorno en test (%)")
    axs[0].legend(loc="upper left")
    fig.suptitle(f"Resultados en test ({N_TEST} series nuevas × 5 semillas; cajas sin outliers)",
                 y=1.03, fontsize=10, color=TINTA)
    guardar(fig, "fig_test.png")


def fig_senales(finales):
    """Por generador: el mejor algoritmo (en validación), la semilla mediana (representativa,
    no la mejor) y la primera serie de test."""
    fig, axs = plt.subplots(4, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, g in enumerate(GENERADORES):
        a = max(ALGORITMOS, key=lambda a: np.mean([r["val"]["logret"].mean() for r in finales[(g, a)]]))
        runs = sorted(finales[(g, a)], key=lambda r: r["val"]["logret"].mean())
        run = runs[len(runs) // 2]
        p = generar_series(GENERADORES[g], 1, 500, SEED_TEST)[0]
        ep = ejecutar_episodio(p, politica_de(run))
        pos, acc = ep["posiciones"], ep["acciones"]
        t = np.arange(5, 5 + len(pos))
        precio = p[5:5 + len(pos)]

        ax = axs[i, 0]
        ax.plot(t, precio, color=TINTA, lw=1.2)
        ax.fill_between(t, precio.min(), precio.max(), where=pos == 1, color=COLOR[a], alpha=0.12, lw=0,
                        label="en posición")
        compras, ventas = (acc == COMPRAR) & (pos == 0), (acc == VENDER) & (pos == 1)
        ax.plot(t[compras], precio[compras], "^", color=POSITIVO, ms=7, label="compra")
        ax.plot(t[ventas], precio[ventas], "v", color=NEGATIVO, ms=7, label="venta")
        ax.set_title(f"{TITULO[g]} — {NOMBRE[a]}: {compras.sum() + ventas.sum()} operaciones")
        ax.set_ylabel("precio")
        if g == "ciclico":
            ax.set_yscale("log")

        ax = axs[i, 1]
        ax.plot(np.arange(5, 5 + len(ep["balances"])), ep["balances"], color=COLOR[a], label=NOMBRE[a])
        for b, pol in (("buy&hold", politica_buy_and_hold), ("aleatorio", politica_aleatoria(0))):
            bal = ejecutar_episodio(p, pol)["balances"]
            ax.plot(np.arange(5, 5 + len(bal)), bal, color=COLOR[b], lw=1.5, ls="--" if b == "buy&hold" else "-",
                    label=NOMBRE[b])
        ax.axhline(1000, color=TINTA_2, lw=1, ls=":")
        ax.set_title("Balance del portafolio ($)")
        if g == "ciclico":
            ax.set_yscale("log")
        if i == 0:
            axs[i, 0].legend(loc="upper left", ncol=3)
            ax.legend(loc="upper left")
    axs[-1, 0].set_xlabel("paso")
    axs[-1, 1].set_xlabel("paso")
    fig.tight_layout()
    guardar(fig, "fig_senales.png")


def tabla(finales, baselines):
    filas = ["| Generador | Agente | Log-ret. test (%) | Mediana balance ($) | Operaciones | "
             "Episodios hasta convergencia | Tiempo (s) | α · γ · ε |",
             "|---|---|---|---|---|---|---|---|"]
    for g in GENERADORES:
        for b in ("aleatorio", "buy&hold"):
            res = baselines[(g, b)]
            lr = [r["logret"].mean() * 100 for r in res]
            bal = np.concatenate([r["balance_final"] for r in res])
            ops = np.concatenate([r["operaciones"] for r in res])
            filas.append(f"| {TITULO[g]} | {NOMBRE[b]} | {np.mean(lr):+.1f} ± {np.std(lr):.1f} | "
                         f"{np.median(bal):,.0f} | {ops.mean():.0f} | — | — | — |")
        for a in ALGORITMOS:
            runs = finales[(g, a)]
            lr = [r["test"]["logret"].mean() * 100 for r in runs]
            bal = np.concatenate([r["test"]["balance_final"] for r in runs])
            ops = np.concatenate([r["test"]["operaciones"] for r in runs])
            curva = np.mean([r["hist"]["eval_logret"] for r in runs], axis=0)
            conv = episodios_convergencia(runs[0]["hist"]["eval_episodio"], curva)
            c = Config(**runs[0]["cfg"])
            alpha = f"{c.alpha}" if c.alpha_decay == 1 else f"{c.alpha}→{c.alpha_min}"
            eps = f"{c.eps_inicial}" if c.eps_inicial == c.eps_final else f"1→0.01 (f={c.frac_exploracion})"
            filas.append(f"| {TITULO[g]} | **{NOMBRE[a]}** | {np.mean(lr):+.1f} ± {np.std(lr):.1f} | "
                         f"{np.median(bal):,.0f} | {ops.mean():.0f} | {conv} | "
                         f"{np.mean([r['tiempo'] for r in runs]):.0f} | {alpha} · {c.gamma} · {eps} |")
    texto = "\n".join(filas)
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, "tabla_resultados.md"), "w", encoding="utf-8") as f:
        f.write(texto + "\n\nLog-ret.: media ± desv. entre semillas de la media sobre las series de test.\n")
    print(texto)


if __name__ == "__main__":
    etapas = {e: cargar(e) for e in ETAPAS}
    finales = ganadores(etapas["epsilon"])
    baselines = cargar("baselines")
    baselines_curva = {(g, b): evaluar(pol, series_curva(g))["logret"].mean()
                       for g in GENERADORES
                       for b, pol in (("buy&hold", politica_buy_and_hold), ("aleatorio", politica_aleatoria(0)))}
    fig_generadores()
    fig_senal_politica(finales)
    fig_ajuste(etapas)
    fig_ajuste({"discretizacion": cargar("discretizacion")}, "fig_discretizacion.png",
               "Discretización de la tendencia sobre la configuración final (validación, 5 semillas; círculo = mejor)")
    fig_convergencia(finales, baselines_curva)
    fig_convergencia_eps(etapas["epsilon"])
    fig_pnl_entrenamiento(finales)
    fig_test(finales, baselines)
    fig_senales(finales)
    tabla(finales, baselines)
