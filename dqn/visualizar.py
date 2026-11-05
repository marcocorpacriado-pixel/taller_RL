"""
Entrenamiento con figuras en vivo, para el notebook.

Desde el directorio dqn, en un notebook:
    from visualizar import mostrar_entrenamiento
    mostrar_entrenamiento("reversion")
"""
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from dqn_agente import COMPRAR, ConfigDQN, MANTENER, VENDER, entrenar
from estilo import COLOR, NEGATIVO, NOMBRE, POSITIVO, TINTA, TINTA_2, TITULO, aplicar_estilo
from experimentos import N_TEST, N_VAL, N_VAL_CURVA, SEED_TEST, SEED_VAL, SEMILLAS
from experimentos_dqn import SEMILLA_LISTA
from trader_rl import (Discretizador, GENERADORES, calcular_tendencias, ejecutar_episodio,
                       evaluar, generar_series, politica_aleatoria, politica_buy_and_hold)

DIR_FIG = Path(__file__).resolve().parent / "figuras"


def _en_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _media_movil(x, n):
    if len(x) < n:
        return np.asarray(x, float)
    return np.convolve(np.asarray(x, float), np.ones(n) / n, mode="valid")


def _mostrar(fig):
    if not _en_notebook():
        return
    from IPython.display import clear_output, display
    clear_output(wait=True)
    display(fig)


def _dibujar_curvas(fig, axs, hist, base_curva, generador, episodio, n_episodios):
    for ax in axs.ravel():
        ax.clear()
    ep = list(hist["eval_episodio"])
    y = np.asarray(hist["eval_logret"], float) * 100
    ax = axs[0, 0]
    ax.plot(ep, y, color=COLOR["dqn"], label=NOMBRE["dqn"])
    for b in ("buy&hold", "aleatorio"):
        ax.axhline(base_curva[b] * 100, color=COLOR[b], ls="--", lw=1.2, label=NOMBRE[b])
    ax.set_title("¿Aprende? Política greedy en validación")
    ax.set_xlabel("episodio")
    ax.set_ylabel("log-retorno (%)")
    ax.legend(loc="best")

    bal = np.log(np.asarray(hist["balance"], float) / 1000.0) * 100
    ventana = 20 if len(bal) >= 20 else 1
    axs[0, 1].plot(np.arange(ventana, len(bal) + 1), _media_movil(bal, ventana), color=COLOR["dqn"])
    axs[0, 1].axhline(0, color=TINTA_2, lw=1)
    titulo_pnl = "Entrenamiento, con exploración" if ventana == 1 else "Entrenamiento, con exploración (media móvil 20)"
    axs[0, 1].set_title(titulo_pnl)
    axs[0, 1].set_xlabel("episodio")
    axs[0, 1].set_ylabel("log-retorno del episodio (%)")

    perd = np.asarray(hist["perdida"], float)
    axs[1, 0].plot(np.arange(1, len(perd) + 1), perd, color=COLOR["dqn"], lw=1.2)
    axs[1, 0].set_title("Pérdida del update (Huber)")
    axs[1, 0].set_xlabel("episodio")
    axs[1, 0].set_ylabel("pérdida")

    eps = np.asarray(hist["epsilon"], float)
    axs[1, 1].plot(np.arange(1, len(eps) + 1), eps, color=TINTA)
    axs[1, 1].set_title("Exploración ε")
    axs[1, 1].set_xlabel("episodio")
    axs[1, 1].set_ylabel("ε")
    axs[1, 1].set_ylim(-0.02, 1.05)

    fig.suptitle(f"{TITULO[generador]} · DQN · episodio {episodio} de {n_episodios}", color=TINTA)
    fig.tight_layout()


def _centros_por_bin(generador, cortes):
    series = generar_series(GENERADORES[generador], 80, 500, seed=123)
    tendencias = np.concatenate([calcular_tendencias(p) for p in series])
    bins = np.digitize(tendencias, cortes)
    centros = []
    for b in range(len(cortes) + 1):
        sel = tendencias[bins == b]
        centros.append(float(np.median(sel)) if len(sel) else 0.0)
    return np.asarray(centros)


def _senal(generador):
    ruta = RAIZ / "resultados" / "senal.pkl"
    if ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)[generador]
    from experimentos import senal_por_bin
    return senal_por_bin(GENERADORES[generador])


def figura_politica(agente, generador):
    """Señal por bin (la misma referencia que el experimento tabular) y decisión del DQN."""
    aplicar_estilo()
    media, se = _senal(generador)
    cortes = Discretizador(GENERADORES[generador], len(media)).cortes
    centros = _centros_por_bin(generador, cortes)
    n_bins = len(media)
    filas = ((0, "fuera: compra"), (1, "dentro: vende"))
    mapa = np.zeros((len(filas), n_bins))
    for i, (pos, _) in enumerate(filas):
        for b, tendencia in enumerate(centros):
            q = agente.valores(np.array([tendencia, pos], np.float32))
            accion, alternativa = (COMPRAR, MANTENER) if pos == 0 else (VENDER, MANTENER)
            d = q[accion] - q[alternativa]
            mapa[i, b] = (d > 0) + 0.5 * (d == 0)

    fig = plt.figure(figsize=(8.2, 4.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax.bar(range(n_bins), media * 100, yerr=2 * se * 100, width=0.75,
           color=[POSITIVO if m > 0 else NEGATIVO for m in media],
           error_kw={"lw": 1, "ecolor": TINTA_2})
    for y in (0.2, -0.2):
        ax.axhline(y, ls="--", lw=1, color=TINTA_2)
    ax.set_xticks([])
    ax.set_ylabel("log-retorno 10 pasos (%)")
    ax.set_title(f"{TITULO[generador]}: señal disponible (arriba) y política DQN (abajo)")

    axp = fig.add_subplot(gs[1])
    axp.imshow(mapa, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, b), v in np.ndenumerate(mapa):
        if v >= 0.5:
            axp.text(b, i, "C" if filas[i][0] == 0 else "V", ha="center", va="center",
                     fontsize=8, color="white" if v > 0.7 else TINTA)
    axp.set_yticks(range(len(filas)), [nombre for _, nombre in filas])
    axp.set_xticks(range(n_bins))
    axp.set_xlabel("bin de tendencia (0 = caída fuerte, 9 = subida fuerte)")
    axp.grid(False)
    return fig


def _texto_test(agente, generador, test, n_episodios, segundos):
    dqn = evaluar(agente.politica_greedy(), test)
    bh = evaluar(politica_buy_and_hold, test)
    aleatorio = [evaluar(politica_aleatoria(s), test) for s in SEMILLAS]
    filas = [("DQN (esta semilla)", [dqn]), ("Buy & Hold", [bh]), ("Aleatorio", aleatorio)]
    lineas = [
        f"{TITULO[generador]} · {n_episodios} episodios · {segundos:.0f} s",
        "Test: 100 series nuevas, semilla 7777 (las mismas que Q-Learning y SARSA).",
        f"{'Agente':<22} {'log-ret. test':>16} {'mediana $':>12} {'operaciones':>12}",
    ]
    for nombre, runs in filas:
        lr = np.array([r["logret"].mean() * 100 for r in runs])
        bal = np.median(np.concatenate([r["balance_final"] for r in runs]))
        ops = np.mean(np.concatenate([r["operaciones"] for r in runs]))
        lr_txt = f"{lr[0]:+.1f} %" if len(lr) == 1 else f"{lr.mean():+.1f} ± {lr.std():.1f} %"
        lineas.append(f"{nombre:<22} {lr_txt:>16} {bal:>12,.0f} {ops:>12.0f}")
    oficial = ConfigDQN().n_episodios
    if n_episodios != oficial:
        lineas.append(f"La tabla oficial entrena {oficial} episodios y promedia {len(SEMILLA_LISTA)} semillas (experimentos_dqn.py).")
    return "\n".join(lineas)


def mostrar_entrenamiento(generador="reversion", n_episodios=None, seed=0, eval_cada=50, **kwargs):
    """Entrena un DQN y va pintando si la política greedy mejora.

    Generadores: ciclico, browniano, reversion, regimenes.
    Si no se pasa n_episodios, usa el de ConfigDQN.
    """
    if generador not in GENERADORES:
        raise KeyError(f"generador debe ser uno de {', '.join(GENERADORES)}")
    aplicar_estilo()
    cfg = ConfigDQN(**kwargs) if n_episodios is None else ConfigDQN(n_episodios=n_episodios, **kwargs)
    gen = GENERADORES[generador]
    val = generar_series(gen, N_VAL, cfg.n_steps, SEED_VAL)
    test = generar_series(gen, N_TEST, cfg.n_steps, SEED_TEST)
    curva = val[:N_VAL_CURVA]
    base_curva = {
        "buy&hold": float(evaluar(politica_buy_and_hold, curva)["logret"].mean()),
        "aleatorio": float(evaluar(politica_aleatoria(0), curva)["logret"].mean()),
    }
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2))
    n_episodios = cfg.n_episodios

    def al_evaluar(episodio, hist, agente):
        _dibujar_curvas(fig, axs, hist, base_curva, generador, episodio, n_episodios)
        _mostrar(fig)

    t0 = time.perf_counter()
    agente, hist = entrenar(cfg, gen, seed, series_validacion=curva, eval_cada=eval_cada, al_evaluar=al_evaluar)
    segundos = time.perf_counter() - t0
    _dibujar_curvas(fig, axs, hist, base_curva, generador, cfg.n_episodios, n_episodios)
    fig_pol = figura_politica(agente, generador)
    texto = _texto_test(agente, generador, test, n_episodios, segundos)

    if _en_notebook():
        from IPython.display import clear_output, display
        clear_output(wait=True)
        display(fig)
        display(fig_pol)
    else:
        DIR_FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(DIR_FIG / f"vista_{generador}.png")
        fig_pol.savefig(DIR_FIG / f"politica_{generador}.png")
        plt.close(fig)
        plt.close(fig_pol)
    print(texto)
    return agente, hist


def pintar_serie(agente, generador, seed_serie=7777):
    """Precio, compras/ventas y balance en la primera serie de test, frente a los baselines."""
    aplicar_estilo()
    precios = generar_series(GENERADORES[generador], 1, 500, seed_serie)[0]
    ep = ejecutar_episodio(precios, agente.politica_greedy())
    pos, acc = ep["posiciones"], ep["acciones"]
    t = np.arange(5, 5 + len(pos))
    precio = precios[5:5 + len(pos)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    ax.plot(t, precio, color=TINTA, lw=1.2)
    ax.fill_between(t, precio.min(), precio.max(), where=pos == 1, color=COLOR["dqn"], alpha=0.12, lw=0)
    compras, ventas = (acc == COMPRAR) & (pos == 0), (acc == VENDER) & (pos == 1)
    ax.plot(t[compras], precio[compras], "^", color=POSITIVO, ms=6, label="compra")
    ax.plot(t[ventas], precio[ventas], "v", color=NEGATIVO, ms=6, label="venta")
    ax.set_title(f"{TITULO[generador]}: {compras.sum() + ventas.sum()} operaciones")
    ax.set_ylabel("precio")
    ax.legend(loc="best")
    if generador == "ciclico":
        ax.set_yscale("log")
        axs[1].set_yscale("log")
    axb = axs[1]
    pasos = np.arange(5, 5 + len(ep["balances"]))
    axb.plot(pasos, ep["balances"], color=COLOR["dqn"], label=NOMBRE["dqn"])
    for nombre, politica, estilo in (
        ("buy&hold", politica_buy_and_hold, "--"),
        ("aleatorio", politica_aleatoria(0), "-"),
    ):
        bal = ejecutar_episodio(precios, politica)["balances"]
        axb.plot(np.arange(5, 5 + len(bal)), bal, color=COLOR[nombre], ls=estilo, lw=1.4, label=NOMBRE[nombre])
    axb.axhline(1000, color=TINTA_2, lw=1, ls=":")
    axb.set_title("Balance ($)")
    axb.legend(loc="best")
    fig.tight_layout()
    if _en_notebook():
        from IPython.display import display
        display(fig)
    else:
        DIR_FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(DIR_FIG / f"serie_{generador}.png")
        plt.close(fig)
    return fig
