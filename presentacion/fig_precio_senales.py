"""
fig_precio_senales.py — Versión para diapositiva de fig_senales/btc_backtest: precio + compras/ventas de Q-Learning
(semilla mediana en validación) en una ventana corta, para que se lean los marcadores. Solo para la presentación.

Uso (desde la raíz del repo):  py presentacion/fig_precio_senales.py   -> presentacion/fig_precio_senales.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt  # noqa: E402  (graficas.py ya fija backend y estilo)
import numpy as np  # noqa: E402

import datos_reales as dr  # noqa: E402
from experimentos import SEED_TEST, cargar, ganadores  # noqa: E402
from graficas import COLOR, NEGATIVO, POSITIVO, TINTA, TITULO, politica_de  # noqa: E402
from trader_rl import COMPRAR, GENERADORES, VENDER, ejecutar_episodio, generar_series  # noqa: E402

A = "qlearning"
VENTANA = slice(100, 300)                       # pasos de la serie sintética que se muestran
BTC_VENTANA = ("2025-01-01", "2025-12-31")     # tramo del test de BTC que se muestra
LECTURA = {"ciclico": "momentum", "reversion": "compra caídas, vende subidas", "regimenes": "sale en los extremos"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_precio_senales.png")


def panel(ax, t, precio, pos, acc, titulo):
    ax.plot(t, precio, color=TINTA, lw=1.3)
    lo, hi = precio.min(), precio.max()
    ax.fill_between(t, lo, hi, where=pos == 1, color=COLOR[A], alpha=0.13, lw=0, step="post", label="en posición")
    compras, ventas = (acc == COMPRAR) & (pos == 0), (acc == VENDER) & (pos == 1)
    ax.plot(t[compras], precio[compras], "^", color=POSITIVO, ms=9, label="compra")
    ax.plot(t[ventas], precio[ventas], "v", color=NEGATIVO, ms=9, label="venta")
    ax.set_title(f"{titulo} · {compras.sum() + ventas.sum()} operaciones", fontsize=12)
    ax.tick_params(labelsize=9)


if __name__ == "__main__":
    finales = ganadores(cargar("epsilon"))
    fig, axs = plt.subplots(2, 2, figsize=(15, 6.6))
    for ax, g in zip(axs.flat, ("ciclico", "reversion", "regimenes")):
        runs = sorted(finales[(g, A)], key=lambda r: r["val"]["logret"].mean())
        p = generar_series(GENERADORES[g], 1, 500, SEED_TEST)[0]           # misma serie que fig_senales.png
        ep = ejecutar_episodio(p, politica_de(runs[len(runs) // 2]))
        t = np.arange(5, 5 + len(ep["posiciones"]))
        precio = p[5:5 + len(t)]
        v = VENTANA
        panel(ax, t[v], precio[v], ep["posiciones"][v], ep["acciones"][v], f"{TITULO[g]}: {LECTURA[g]}")
        ax.set_xlabel("paso", fontsize=9)

    # BTC: mismo cálculo que datos_reales.fig_backtest (test, Q-Learning, semilla mediana)
    serie = dr.tramo("test")
    ep = dr.semilla_mediana(dr.seleccion(cargar("btc")["resultados"])[A])["test"]
    fechas, precio = serie.index[5:-1], serie.values[5:-1]
    m = (fechas >= BTC_VENTANA[0]) & (fechas <= BTC_VENTANA[1])
    assert len(ep["acciones"]) == len(fechas)
    ax = axs[1, 1]
    panel(ax, fechas[m], precio[m], ep["posiciones"][m], ep["acciones"][m], "BTC-USD real, 2025")
    ax.set_ylabel("USD", fontsize=9)
    ax.tick_params(axis="x", labelrotation=0)

    axs[0, 0].legend(loc="upper left", ncol=3, fontsize=10)
    for ax in axs[:, 0]:
        ax.set_ylabel(ax.get_ylabel() or "precio", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(OUT)
