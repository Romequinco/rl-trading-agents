"""Estilo común de las figuras: misma paleta y formato en todos los notebooks y en el informe.

Uso: `from src.style import setup, AGENT_COLORS, MARKET_COLORS, savefig` y `setup()` al principio.
"""

import matplotlib.pyplot as plt

from .experiment import RESULTS_DIR

AGENT_COLORS = {
    "aleatorio": "#9AA0A6",
    "buy_hold": "#F29D38",
    "qlearning": "#2B6CB0",
    "sarsa": "#2F9E6E",
}
MARKET_COLORS = {
    "tendencia_ciclo": "#6C5CE7",
    "browniano": "#0984E3",
    "reversion_media": "#00A884",
    "regimenes": "#D63031",
}
MARKET_LABELS = {
    "tendencia_ciclo": "1 · Tendencia + ciclo",
    "browniano": "2 · Browniano geométrico",
    "reversion_media": "3 · Reversión a la media",
    "regimenes": "4 · Regímenes",
}
POSITION_COLORS = {1: "#2F9E6E", 0: "#D63031"}  # con activo / sin activo, como en el taller


def setup():
    """Aplica el estilo común de matplotlib."""
    plt.rcParams.update({
        "figure.figsize": (11, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })


def savefig(fig, name):
    """Guarda la figura en `results/figures/<name>.png` (lo que se use en el informe)."""
    path = RESULTS_DIR / "figures" / f"{name}.png"
    fig.savefig(path)
    return path
