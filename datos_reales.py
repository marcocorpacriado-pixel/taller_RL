"""
datos_reales.py — Q-Learning y SARSA sobre datos reales de BTC-USD (velas diarias, yfinance).

Uso:
    py datos_reales.py descargar    descarga la serie y la congela en datos/btc_usd.csv
    py datos_reales.py              entrena el grid, evalúa y genera figuras (usa el CSV)
    py datos_reales.py figuras      sólo figuras y tabla, a partir de resultados/btc.pkl

Diseño (BITACORA, sección 4):
  * Serie congelada en CSV: yfinance puede limitar peticiones o revisar datos antiguos.
  * División CRONOLÓGICA, sin barajar: entrenamiento 2014-09 → 2020, validación
    2021–2022, test 2023 → hoy. Barajar fechas filtraría información del futuro.
  * Cada episodio de entrenamiento es una ventana aleatoria de 500 días del periodo
    de entrenamiento (generador_ventanas). Tiene la misma firma que los generadores
    sintéticos, así que entrenar() y Discretizador se reutilizan tal cual y los
    cuantiles salen SÓLO de datos de entrenamiento.
  * Validación y test: backtest continuo de todo el periodo.
  * Grid α × γ por algoritmo, 5 semillas; selección por log-retorno en validación.
"""
import os
import sys
import time
from dataclasses import asdict, replace
from functools import lru_cache
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimentos import ALGORITMOS, BASE, SEMILLAS, cargar, guardar, media_por_bin
from graficas import DIR as DIR_FIGURAS
from graficas import COLOR, NEGATIVO, NOMBRE, POSITIVO, TINTA, TINTA_2, guardar as guardar_figura
from trader_rl import (COMPRAR, MANTENER, VENDER, Discretizador, ejecutar_episodio, entrenar,
                       politica_aleatoria, politica_buy_and_hold)

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "btc_usd.csv")
TICKER = "BTC-USD"
PERIODOS = {"train": ("2014-09-17", "2020-12-31"),
            "val": ("2021-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2026-10-26")}
GRID = [dict(alpha=a, gamma=g) for a in (0.001, 0.002, 0.005, 0.01) for g in (0.9, 0.95, 0.99)]
DIAS_ANYO = 365  # cripto cotiza todos los días


# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
def descargar():
    import yfinance as yf
    fin = pd.Timestamp(PERIODOS["test"][1]) + pd.Timedelta(days=1)  # `end` es exclusivo
    df = yf.download(TICKER, start=PERIODOS["train"][0], end=fin, interval="1d",
                     auto_adjust=True, progress=False)
    cierre = df["Close"][TICKER].dropna().rename("close")
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    cierre.to_csv(CSV, index_label="date")
    print(f"{len(cierre)} días guardados en {CSV} ({cierre.index[0].date()} a {cierre.index[-1].date()})")


@lru_cache(maxsize=1)
def precios():
    return pd.read_csv(CSV, index_col="date", parse_dates=True)["close"]


def tramo(nombre):
    """Precios de un periodo (pd.Series con fechas).

    A validación y test se les anteponen los 5 días previos: el entorno necesita
    6 precios para la primera tendencia y así se decide desde el primer día.
    """
    s = precios()
    ini, fin = PERIODOS[nombre]
    idx = np.flatnonzero((s.index >= ini) & (s.index <= fin))
    inicio = idx[0] if nombre == "train" else idx[0] - 5
    return s.iloc[inicio: idx[-1] + 1]


def generador_ventanas(serie):
    """Convierte una serie real en un "generador": cada llamada devuelve una ventana
    aleatoria de n_steps días (usa el RNG global de numpy, como los sintéticos).

    ponytail: las ventanas se solapan mucho (≈ 2300 días de entrenamiento), así que
    el agente ve los mismos datos muchas veces; más historia o más activos lo mitigarían.
    """
    def gen(n_steps=500):
        i = np.random.randint(0, len(serie) - n_steps + 1)
        return serie[i: i + n_steps]
    return gen


# -----------------------------------------------------------------------------
# Métricas financieras de un backtest
# -----------------------------------------------------------------------------
def metricas(ep):
    b, pos, acc = ep["balances"], ep["posiciones"], ep["acciones"]
    r = b[1:] / b[:-1] - 1                        # retornos diarios de la cartera
    anyos = len(r) / DIAS_ANYO
    return {
        "Rent. total (%)": (b[-1] / b[0] - 1) * 100,
        "CAGR (%)": ((b[-1] / b[0]) ** (1 / anyos) - 1) * 100,
        "Volatilidad anual (%)": r.std() * np.sqrt(DIAS_ANYO) * 100,
        "Sharpe": r.mean() / r.std() * np.sqrt(DIAS_ANYO) if r.std() > 0 else 0.0,
        "Máx. drawdown (%)": (1 - b / np.maximum.accumulate(b)).max() * 100,
        "Operaciones": np.sum((acc == COMPRAR) & (pos == 0)) + np.sum((acc == VENDER) & (pos == 1)),
        "Tiempo invertido (%)": pos.mean() * 100,
    }


def logret(ep):
    return np.log(ep["balances"][-1] / ep["balances"][0])


# -----------------------------------------------------------------------------
# Entrenamiento
# -----------------------------------------------------------------------------
def entrenar_y_evaluar(tarea):
    cambios, seed = tarea
    cfg = replace(BASE, **cambios)
    gen = generador_ventanas(tramo("train").values)
    disc = Discretizador(gen, cfg.n_bins)
    val, test = tramo("val").values, tramo("test").values
    t0 = time.perf_counter()
    agente, hist = entrenar(cfg, gen, disc, seed, series_validacion=[val], eval_cada=250)
    politica = agente.politica_greedy(disc)
    return {"cfg": asdict(cfg), "seed": seed, "Q": agente.Q, "cortes": disc.cortes, "hist": hist,
            "tiempo": time.perf_counter() - t0,
            "val": ejecutar_episodio(val, politica), "test": ejecutar_episodio(test, politica)}


def ejecutar():
    tareas = [(dict(algoritmo=a, **g), s) for a in ALGORITMOS for g in GRID for s in SEMILLAS]
    print(f"[btc] {len(tareas)} entrenamientos...", flush=True)
    t0 = time.perf_counter()
    with Pool(min(len(tareas), os.cpu_count())) as pool:
        resultados = pool.map(entrenar_y_evaluar, tareas)
    baselines = {p: {"buy&hold": [ejecutar_episodio(tramo(p).values, politica_buy_and_hold)],
                     "aleatorio": [ejecutar_episodio(tramo(p).values, politica_aleatoria(s)) for s in SEMILLAS]}
                 for p in ("val", "test")}
    guardar("btc", {"resultados": resultados, "baselines": baselines})
    print(f"[btc] terminado en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def seleccion(resultados):
    """algoritmo -> las 5 semillas de la config (α, γ) con mejor log-retorno medio en validación."""
    grupos = {}
    for r in resultados:
        c = r["cfg"]
        grupos.setdefault((c["algoritmo"], c["alpha"], c["gamma"]), []).append(r)
    mejores = {}
    for (a, _, _), runs in grupos.items():
        puntuacion = np.mean([logret(r["val"]) for r in runs])
        if a not in mejores or puntuacion > mejores[a][0]:
            mejores[a] = (puntuacion, runs)
    return {a: runs for a, (_, runs) in mejores.items()}


def semilla_mediana(runs):
    """Semilla representativa: la mediana en VALIDACIÓN (no la mejor en test)."""
    return sorted(runs, key=lambda r: logret(r["val"]))[len(runs) // 2]


# -----------------------------------------------------------------------------
# Figuras y tabla
# -----------------------------------------------------------------------------
def fig_precio():
    fig, ax = plt.subplots(figsize=(12, 3))
    s = precios()
    ax.plot(s.index, s.values, color=TINTA, lw=1.2)
    for (nombre, (ini, fin)), c in zip(PERIODOS.items(), ("#b5b4ae", COLOR["buy&hold"], COLOR["sarsa"])):
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color=c, alpha=0.12, lw=0)
        ax.text(pd.Timestamp(ini), s.max(), f" {nombre}", va="top", color=TINTA_2, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("precio (USD, log)")
    ax.set_title(f"{TICKER} diario: división cronológica entrenamiento / validación / test")
    guardar_figura(fig, "btc_precio.png")


def fig_ajuste(resultados, elegidos):
    alphas = sorted({g["alpha"] for g in GRID})
    gammas = sorted({g["gamma"] for g in GRID})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    mapas = {}
    for a in ALGORITMOS:
        m = np.zeros((len(alphas), len(gammas)))
        for r in resultados:
            c = r["cfg"]
            if c["algoritmo"] == a:
                m[alphas.index(c["alpha"]), gammas.index(c["gamma"])] += logret(r["val"]) * 100 / len(SEMILLAS)
        mapas[a] = m
    lim = max(np.abs(m).max() for m in mapas.values())
    for ax, a in zip(axs, ALGORITMOS):
        ax.imshow(mapas[a], cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(mapas[a]):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * lim else TINTA)
        c = elegidos[a][0]["cfg"]
        ax.add_patch(plt.Rectangle((gammas.index(c["gamma"]) - 0.5, alphas.index(c["alpha"]) - 0.5), 1, 1,
                                   fill=False, ec=TINTA, lw=2))
        ax.set_xticks(range(len(gammas)), [f"γ={g}" for g in gammas])
        ax.set_yticks(range(len(alphas)), [f"α={x}" for x in alphas])
        ax.set_title(NOMBRE[a])
        ax.grid(False)
    fig.suptitle("Log-retorno en validación 2021–2022 (%), media de 5 semillas; recuadro = elegido",
                 y=1.02, fontsize=10, color=TINTA)
    guardar_figura(fig, "btc_ajuste.png")


def fig_senal_politica(elegidos):
    """Señal por bin en entrenamiento y en test (¿es estable?) frente a la política aprendida."""
    cortes = elegidos[ALGORITMOS[0]][0]["cortes"]
    n_bins = len(cortes) + 1
    fig = plt.figure(figsize=(9, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.7], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    x = np.arange(n_bins)
    for desplaz, periodo, color in ((-0.2, "train", "#8a8984"), (0.2, "test", COLOR["sarsa"])):
        media, se = media_por_bin([tramo(periodo).values], cortes)
        ax.bar(x + desplaz, media * 100, yerr=2 * se * 100, width=0.38, color=color,
               error_kw={"lw": 1, "ecolor": TINTA_2}, label=f"{periodo} ({PERIODOS[periodo][0][:4]}–{PERIODOS[periodo][1][:4]})")
    for y in (0.2, -0.2):
        ax.axhline(y, ls="--", lw=1, color=TINTA_2)
    ax.set_xticks([])
    ax.set_ylabel("log-retorno 10 días (%)")
    ax.legend(loc="upper left")
    ax.set_title(f"{TICKER}: señal por bin de tendencia (±2 e.s.) y política aprendida")

    filas = [(a, pos) for a in ALGORITMOS for pos in (0, 1)]
    mapa = np.zeros((len(filas), n_bins))
    for i, (a, pos) in enumerate(filas):
        for r in elegidos[a]:
            d = r["Q"][:, pos, COMPRAR if pos == 0 else VENDER] - r["Q"][:, pos, MANTENER]
            mapa[i] += ((d > 0) + 0.5 * (d == 0)) / len(elegidos[a])
    axp = fig.add_subplot(gs[1])
    axp.imshow(mapa, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, b), v in np.ndenumerate(mapa):
        if v >= 0.5:
            axp.text(b, i, "C" if filas[i][1] == 0 else "V", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.7 else TINTA)
    axp.set_yticks(range(len(filas)), [f"{NOMBRE[a]} · {'fuera: compra' if p == 0 else 'dentro: vende'}"
                                      for a, p in filas])
    axp.set_xticks(x)
    axp.set_xlabel("bin de tendencia (cuantiles de entrenamiento; 0 = caída fuerte, 9 = subida fuerte)")
    axp.grid(False)
    guardar_figura(fig, "btc_senal_politica.png")


def fig_backtest(elegidos, baselines):
    fig, axs = plt.subplots(2, 2, figsize=(15, 7), sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    for j, periodo in enumerate(("val", "test")):
        serie = tramo(periodo)
        fechas = serie.index[5:]
        ax_p, ax_b = axs[0, j], axs[1, j]
        ax_p.plot(fechas, serie.values[5:], color=TINTA, lw=1.1)
        for a in ALGORITMOS:
            ep = semilla_mediana(elegidos[a])[periodo]
            ax_b.plot(fechas, ep["balances"], color=COLOR[a], label=f"{NOMBRE[a]} (semilla mediana)")
            if a == ALGORITMOS[0]:
                pos, acc = ep["posiciones"], ep["acciones"]
                f, p = fechas[:-1], serie.values[5:-1]
                compras, ventas = (acc == COMPRAR) & (pos == 0), (acc == VENDER) & (pos == 1)
                ax_p.plot(f[compras], p[compras], "^", color=POSITIVO, ms=6, label=f"compra ({NOMBRE[a]})")
                ax_p.plot(f[ventas], p[ventas], "v", color=NEGATIVO, ms=6, label=f"venta ({NOMBRE[a]})")
        ax_b.plot(fechas, baselines[periodo]["buy&hold"][0]["balances"], color=COLOR["buy&hold"], ls="--",
                  label=NOMBRE["buy&hold"])
        ax_b.plot(fechas, baselines[periodo]["aleatorio"][0]["balances"], color=COLOR["aleatorio"], lw=1.5,
                  label=NOMBRE["aleatorio"])
        ax_b.axhline(1000, color=TINTA_2, lw=1, ls=":")
        ax_p.set_yscale("log")
        ax_p.set_title(f"{'Validación' if periodo == 'val' else 'Test'} {PERIODOS[periodo][0][:4]}–"
                       f"{PERIODOS[periodo][1][:4]}: precio y señales")
        ax_b.set_title("Balance del portafolio ($)")
    axs[0, 0].set_ylabel("precio (USD, log)")
    axs[1, 0].set_ylabel("balance ($)")
    axs[0, 0].legend(loc="upper left")
    axs[1, 0].legend(loc="upper left")
    fig.tight_layout()
    guardar_figura(fig, "btc_backtest.png")


def tabla(elegidos, baselines):
    claves = list(metricas(baselines["test"]["buy&hold"][0]))
    filas = []
    for periodo in ("val", "test"):
        filas += [f"\n**{'Validación' if periodo == 'val' else 'Test'} "
                  f"({PERIODOS[periodo][0]} a {PERIODOS[periodo][1]})**\n",
                  "| Agente | " + " | ".join(claves) + " |", "|---" * (len(claves) + 1) + "|"]
        agentes = [(NOMBRE["buy&hold"], baselines[periodo]["buy&hold"]),
                   (NOMBRE["aleatorio"], baselines[periodo]["aleatorio"])]
        for a in ALGORITMOS:
            c = elegidos[a][0]["cfg"]
            agentes.append((f"**{NOMBRE[a]}** (α={c['alpha']}, γ={c['gamma']})", [r[periodo] for r in elegidos[a]]))
        for nombre, episodios in agentes:
            m = [metricas(ep) for ep in episodios]
            celdas = []
            for k in claves:
                v = np.array([x[k] for x in m], dtype=float)
                fmt = "{:.2f}" if k == "Sharpe" else "{:.1f}"
                celdas.append(fmt.format(v.mean()) if len(v) == 1 else f"{fmt.format(v.mean())} ± {fmt.format(v.std())}")
            filas.append(f"| {nombre} | " + " | ".join(celdas) + " |")
    texto = "\n".join(filas) + "\n\nAgentes RL y aleatorio: media ± desv. entre 5 semillas.\n"
    with open(os.path.join(DIR_FIGURAS, "btc_tabla.md"), "w", encoding="utf-8") as f:
        f.write(f"# {TICKER} — métricas de backtest\n" + texto)
    print(texto)


def figuras():
    datos = cargar("btc")
    elegidos = seleccion(datos["resultados"])
    fig_precio()
    fig_ajuste(datos["resultados"], elegidos)
    fig_senal_politica(elegidos)
    fig_backtest(elegidos, datos["baselines"])
    tabla(elegidos, datos["baselines"])


if __name__ == "__main__":
    for paso in sys.argv[1:] or ["entrenar", "figuras"]:
        {"descargar": descargar, "entrenar": ejecutar, "figuras": figuras}[paso]()
