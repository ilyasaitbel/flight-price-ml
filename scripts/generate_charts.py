"""
Generate the figures used by the README from the real dataset and the trained model.

Usage
-----
    python scripts/generate_charts.py

Outputs (into docs/)
------------------------
    price_distribution.png   price histogram, overall and split by travel class
    booking_curve.png        average fare vs. days before departure
    route_heatmap.png        average fare for every city pair
    feature_importance.png   aggregated feature importance from the fitted pipeline
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Clean_Dataset.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "flight_price_pipeline.joblib"
DOCS_DIR = BASE_DIR / "docs"

# =========================================================
# THEME
# =========================================================

BG = "#0B1120"
PANEL = "#111827"
GRID = "#1F2937"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
BLUE = "#60A5FA"
INDIGO = "#818CF8"
EMERALD = "#34D399"
AMBER = "#FBBF24"
ROSE = "#FB7185"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "figure.dpi": 140,
    }
)


def style_axes(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, color=TEXT, pad=32 if subtitle else 10, loc="left")
    if subtitle:
        ax.text(
            0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=10,
            va="bottom",
        )
    ax.grid(axis="y", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def rupees(x, _pos=None) -> str:
    if x >= 1000:
        return f"₹{x/1000:,.0f}k"
    return f"₹{x:,.0f}"


def chart_price_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(df["price"], bins=90, color=BLUE, edgecolor="none", alpha=0.9)
    ax.axvline(df["price"].median(), color=AMBER, lw=2, ls="--")
    ax.annotate(
        f"median  ₹{df['price'].median():,.0f}",
        xy=(df["price"].median(), ax.get_ylim()[1] * 0.92),
        xytext=(12, 0),
        textcoords="offset points",
        color=AMBER,
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Fare (INR)")
    ax.set_ylabel("Flights")
    ax.xaxis.set_major_formatter(FuncFormatter(rupees))
    style_axes(
        ax,
        "Fare distribution",
        f"{len(df):,} flights  ·  ₹{df['price'].min():,.0f} to ₹{df['price'].max():,.0f}",
    )

    ax = axes[1]
    colors = {"Economy": BLUE, "Business": ROSE}
    for klass, colour in colors.items():
        subset = df.loc[df["class"] == klass, "price"]
        ax.hist(
            subset,
            bins=90,
            color=colour,
            edgecolor="none",
            alpha=0.75,
            label=f"{klass}  (n={len(subset):,})",
        )
    ax.set_xlabel("Fare (INR)")
    ax.set_ylabel("Flights")
    ax.xaxis.set_major_formatter(FuncFormatter(rupees))
    legend = ax.legend(frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(TEXT)
    style_axes(
        ax,
        "Fare by travel class",
        "Business class shifts the whole distribution, not just its tail",
    )

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "price_distribution.png", bbox_inches="tight")
    plt.close(fig)


def chart_booking_curve(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, klass, colour in ((axes[0], "Economy", BLUE), (axes[1], "Business", ROSE)):
        subset = df[df["class"] == klass]
        grouped = subset.groupby("days_left")["price"]
        mean = grouped.mean()
        sem = grouped.sem()
        low, high = mean - 1.96 * sem, mean + 1.96 * sem

        ax.plot(mean.index, mean.values, color=colour, lw=2.4)
        ax.fill_between(mean.index, low.values, high.values, color=colour, alpha=0.16)
        ax.set_xlabel("Days before departure")
        ax.set_title(f"{klass}", color=TEXT, fontsize=13, pad=10, loc="left")
        ax.grid(axis="y", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.yaxis.set_major_formatter(FuncFormatter(rupees))

        first, last = mean.iloc[0], mean.iloc[-1]
        ax.annotate(
            f"{last:,.0f}",
            xy=(mean.index[-1], last),
            xytext=(-6, 12),
            textcoords="offset points",
            color=colour,
            fontsize=11,
            fontweight="bold",
            ha="right",
        )

    fig.suptitle(
        "Book earlier, pay less",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        x=0.055,
        ha="left",
        y=1.12,
    )
    fig.text(
        0.055,
        1.04,
        "Average fare against days remaining before departure, with a 95% confidence band",
        color=MUTED,
        fontsize=10,
        ha="left",
    )
    axes[0].set_ylabel("Average fare (INR)")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "booking_curve.png", bbox_inches="tight")
    plt.close(fig)


def chart_route_heatmap(df: pd.DataFrame) -> None:
    cities = sorted(df["source_city"].unique())
    grid = df.pivot_table(
        index="source_city",
        columns="destination_city",
        values="price",
        aggfunc="mean",
    ).reindex(index=cities, columns=cities)

    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    mesh = ax.imshow(grid.values, cmap="magma", aspect="auto")

    ax.set_xticks(range(len(cities)), cities, rotation=35, ha="right")
    ax.set_yticks(range(len(cities)), cities)
    ax.set_title(
        "Average fare by route",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=32,
    )
    ax.text(
        0,
        1.015,
        "Every city pair, mean price across all airlines, stops and booking windows",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    finite = grid.values[np.isfinite(grid.values)]
    vmax = np.nanpercentile(finite, 98)
    # Pick the label colour from the actual colour the cell was painted with,
    # so text stays legible on both the pale and the near-black end of the map.
    cmap = matplotlib.colormaps["magma"]
    for i in range(len(cities)):
        for j in range(len(cities)):
            value = grid.values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                continue
            rgba = cmap((value - np.nanmin(grid.values)) / (np.nanmax(grid.values) - np.nanmin(grid.values)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                j,
                i,
                f"{value/1000:,.1f}k",
                ha="center",
                va="center",
                fontsize=10,
                color="#0B1120" if luminance > 0.55 else TEXT,
                fontweight="bold",
            )

    bar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.03)
    bar.set_label("Mean fare (INR)", color=MUTED, fontsize=10)
    bar.ax.tick_params(colors=MUTED, length=0)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "route_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def chart_feature_importance() -> None:
    if not MODEL_PATH.exists():
        print(f"  ! {MODEL_PATH.name} missing - skipping feature importance chart")
        return

    pipeline = joblib.load(MODEL_PATH)
    encoded = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["regressor"].feature_importances_

    # Collapse one-hot columns back to their parent feature. Matching is done on
    # the "<feature>_<value>" prefix against the real feature names, because
    # category values themselves contain underscores ("Early_Morning",
    # "two_or_more") and naive splitting would mangle them.
    categorical = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
    ]
    numerical = ["duration", "days_left"]

    totals: dict[str, float] = {name: 0.0 for name in categorical + numerical}
    for name, value in zip(encoded, importances):
        bare = name.split("__", 1)[1]
        parent = next(
            (f for f in categorical + numerical if bare == f or bare.startswith(f + "_")),
            None,
        )
        if parent is None:
            continue
        totals[parent] += float(value)

    ordered = sorted(totals.items(), key=lambda kv: kv[1])
    names = [k.replace("_", " ") for k, _ in ordered]
    values = [v * 100 for _, v in ordered]

    colours = [BLUE if v >= 5 else INDIGO for v in values]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    bars = ax.barh(names, values, color=colours, height=0.66)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=10,
            color=MUTED,
        )

    ax.set_xlabel("Share of total importance (%)")
    ax.set_xlim(0, max(values) * 1.22)
    style_axes(
        ax,
        "What actually drives the fare",
        "Aggregated feature importance from the fitted Random Forest",
    )
    ax.grid(axis="x", alpha=0.35, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    print("Generating charts ...")
    chart_price_distribution(df)
    print("  + price_distribution.png")
    chart_booking_curve(df)
    print("  + booking_curve.png")
    chart_route_heatmap(df)
    print("  + route_heatmap.png")
    chart_feature_importance()
    print("  + feature_importance.png")
    print("Done ->", DOCS_DIR.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
