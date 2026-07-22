"""Shared plotting style."""
import matplotlib as mpl

C_FCC = "#0072B2"
C_IADC = "#D55E00"


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
