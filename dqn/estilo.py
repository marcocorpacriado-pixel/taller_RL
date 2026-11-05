"""Colores y estilo de las figuras, alineados con graficas.py del experimento tabular."""
import matplotlib.pyplot as plt

COLOR = {
    "qlearning": "#2a78d6",
    "sarsa": "#eb6834",
    "buy&hold": "#1baf7a",
    "aleatorio": "#8a8984",
    "dqn": "#6d28d9",
}
NOMBRE = {
    "qlearning": "Q-Learning",
    "sarsa": "SARSA",
    "buy&hold": "Buy & Hold",
    "aleatorio": "Aleatorio",
    "dqn": "DQN",
}
TITULO = {
    "ciclico": "Tendencia + ciclo",
    "browniano": "Browniano geométrico",
    "reversion": "Reversión a la media",
    "regimenes": "Regímenes alcista/bajista",
}
TINTA, TINTA_2, POSITIVO, NEGATIVO = "#0b0b0b", "#52514e", "#2a78d6", "#e34948"
GENERADORES_ORDEN = ("ciclico", "browniano", "reversion", "regimenes")


def aplicar_estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
        "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
    })
