import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Data from Table: Comparative Results
# ============================================================

comparative_data = {
    "Complexity": {
        "Pretrained": {
            "Output Positive Rate": 0.00,
            "Parseable Output Rate": 22.16,
            "Centroid Distance": 36.144,
            "Cosine Similarity": 0.913,
        },
        "Steering": {
            "Output Positive Rate": 0.54,
            "Parseable Output Rate": 18.92,
            "Centroid Distance": 33.995,
            "Cosine Similarity": 0.925,
        },
        "Permanent": {
            "Output Positive Rate": 0.00,
            "Parseable Output Rate": 0.00,
            "Centroid Distance": 53.946,
            "Cosine Similarity": 0.937,
        },
    },

    "Coupling": {
        "Pretrained": {
            "Output Positive Rate": 0.00,
            "Parseable Output Rate": 22.16,
            "Centroid Distance": 7.214,
            "Cosine Similarity": 0.878,
        },
        "Steering": {
            "Output Positive Rate": 0.54,
            "Parseable Output Rate": 17.30,
            "Centroid Distance": 6.556,
            "Cosine Similarity": 0.893,
        },
        "Permanent": {
            "Output Positive Rate": 2.70,
            "Parseable Output Rate": 86.49,
            "Centroid Distance": 2.690,
            "Cosine Similarity": 0.952,
        },
    },

    "Modularity": {
        "Pretrained": {
            "Output Positive Rate": 1.08,
            "Parseable Output Rate": 22.16,
            "Centroid Distance": 20.855,
            "Cosine Similarity": 0.900,
        },
        "Steering": {
            "Output Positive Rate": 2.70,
            "Parseable Output Rate": 25.95,
            "Centroid Distance": 16.875,
            "Cosine Similarity": 0.925,
        },
        "Permanent": {
            "Output Positive Rate": 5.41,
            "Parseable Output Rate": 85.95,
            "Centroid Distance": 3.101,
            "Cosine Similarity": 0.995,
        },
    },
}


# ============================================================
# Data from Table: Comparative Capability
# ============================================================

capability_data = {
    "Complexity": {
        "Permanent Rate": 0.00,
        "Steering Rate": 0.54,
        "Permanent Parseable": 0.00,
        "Steering Parseable": 18.92,
        "Permanent NLL Change": -4.53,
        "Steering NLL Change": 2.21,
    },
    "Coupling": {
        "Permanent Rate": 2.70,
        "Steering Rate": 0.54,
        "Permanent Parseable": 86.49,
        "Steering Parseable": 17.30,
        "Permanent NLL Change": -4.22,
        "Steering NLL Change": 0.34,
    },
    "Modularity": {
        "Permanent Rate": 5.41,
        "Steering Rate": 2.70,
        "Permanent Parseable": 85.95,
        "Steering Parseable": 25.95,
        "Permanent NLL Change": 1.78,
        "Steering NLL Change": 5.67,
    },
}


# ============================================================
# Figure 1: Radar Charts
# ============================================================

metrics = [
    "Output Positive Rate",
    "Parseable Output Rate",
    "Centroid Distance",
    "Cosine Similarity",
]

models = ["Pretrained", "Permanent", "Steering"]


def normalize_property(property_data):
    """
    Normalize each metric to [0, 1] within one property.

    Higher is better for:
        Output Positive Rate
        Parseable Output Rate
        Cosine Similarity

    Lower is better for:
        Centroid Distance
    """

    normalized = {
        model: {}
        for model in models
    }

    for metric in metrics:

        values = np.array(
            [property_data[model][metric] for model in models],
            dtype=float
        )

        min_value = values.min()
        max_value = values.max()

        if np.isclose(max_value, min_value):
            scores = np.ones(len(values))
        else:
            scores = (
                (values - min_value)
                / (max_value - min_value)
            )

        # Lower centroid distance is better.
        if metric == "Centroid Distance":
            scores = 1.0 - scores

        for model, score in zip(models, scores):
            normalized[model][metric] = score

    return normalized


def create_radar_figure():

    number_of_metrics = len(metrics)

    angles = np.linspace(
        0,
        2 * np.pi,
        number_of_metrics,
        endpoint=False
    ).tolist()

    # Close the polygon.
    angles += angles[:1]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4.5),
        subplot_kw={"polar": True}
    )

    for ax, (property_name, property_data) in zip(
        axes,
        comparative_data.items()
    ):

        normalized = normalize_property(property_data)

        for model in models:

            values = [
                normalized[model][metric]
                for metric in metrics
            ]

            values += values[:1]

            ax.plot(
                angles,
                values,
                linewidth=2,
                marker="o",
                label=model
            )

            ax.fill(
                angles,
                values,
                alpha=0.08
            )

        labels = [
            "Output\nPositive Rate",
            "Parseable\nOutput Rate",
            "Centroid\nProximity",
            "Cosine\nSimilarity",
        ]

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            labels,
            fontsize=9
        )

        ax.set_ylim(0, 1)

        ax.set_yticks([
            0.25,
            0.50,
            0.75,
            1.00
        ])

        ax.set_yticklabels([
            "0.25",
            "0.50",
            "0.75",
            "1.00"
        ], fontsize=7)

        ax.set_title(
            property_name,
            fontsize=12,
            fontweight="bold",
            pad=18
        )

    # One common legend for all three radar charts.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    plt.savefig(
        "comparative_radar.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        "comparative_radar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Figure 2: Output Positive Rate Bar Chart
# ============================================================

def create_output_rate_bar_figure():

    properties = list(capability_data.keys())
    permanent = [capability_data[p]["Permanent Rate"] for p in properties]
    steering = [capability_data[p]["Steering Rate"] for p in properties]
    x = np.arange(len(properties))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    permanent_bars = ax.bar(x - width / 2, permanent, width, label="Permanent")
    steering_bars = ax.bar(x + width / 2, steering, width, label="Steering")

    ax.set_xticks(x)
    ax.set_xticklabels(properties, fontsize=10)
    ax.set_ylabel("Output Positive Rate (%)", fontsize=10)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in (permanent_bars, steering_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08,
                    f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(permanent + steering) * 1.18 + 0.1)
    plt.tight_layout()
    plt.savefig("comparative_output_positive_rate.pdf", bbox_inches="tight")
    plt.savefig("comparative_output_positive_rate.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Figure 3: Parseable Output Rate Bar Chart
# ============================================================

def create_parseable_rate_bar_figure():

    properties = list(capability_data.keys())
    permanent = [capability_data[p]["Permanent Parseable"] for p in properties]
    steering = [capability_data[p]["Steering Parseable"] for p in properties]
    x = np.arange(len(properties))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    permanent_bars = ax.bar(x - width / 2, permanent, width, label="Permanent")
    steering_bars = ax.bar(x + width / 2, steering, width, label="Steering")

    ax.set_xticks(x)
    ax.set_xticklabels(properties, fontsize=10)
    ax.set_ylabel("Parseable Output Rate (%)", fontsize=10)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars in (permanent_bars, steering_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0,
                    f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig("comparative_parseable_output_rate.pdf", bbox_inches="tight")
    plt.savefig("comparative_parseable_output_rate.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Figure 4: Diverging Horizontal Bar Chart
# ============================================================

def create_nll_bar_figure():

    properties = list(capability_data.keys())

    permanent_changes = [
        capability_data[p]["Permanent NLL Change"]
        for p in properties
    ]

    steering_changes = [
        capability_data[p]["Steering NLL Change"]
        for p in properties
    ]

    y = np.arange(len(properties))

    bar_height = 0.32

    fig, ax = plt.subplots(
        figsize=(7.5, 4.2)
    )

    permanent_bars = ax.barh(
        y - bar_height / 2,
        permanent_changes,
        height=bar_height,
        label="Permanent"
    )

    steering_bars = ax.barh(
        y + bar_height / 2,
        steering_changes,
        height=bar_height,
        label="Steering"
    )

    # Zero reference line.
    ax.axvline(
        0,
        linewidth=1.2
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        properties,
        fontsize=10
    )

    ax.set_xlabel(
        "Reference-patch NLL Change (%)",
        fontsize=10
    )

    ax.invert_yaxis()

    ax.legend(
        frameon=False,
        loc="lower right"
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    def add_labels(bars):

        for bar in bars:

            value = bar.get_width()

            y_position = (
                bar.get_y()
                + bar.get_height() / 2
            )

            if value >= 0:
                x_position = value + 0.12
                alignment = "left"
            else:
                x_position = value - 0.12
                alignment = "right"

            ax.text(
                x_position,
                y_position,
                f"{value:+.2f}%",
                va="center",
                ha=alignment,
                fontsize=9
            )

    add_labels(permanent_bars)
    add_labels(steering_bars)

    # Symmetric scale makes positive/negative changes
    # visually comparable.
    maximum = max(
        max(abs(v) for v in permanent_changes),
        max(abs(v) for v in steering_changes)
    )

    limit = np.ceil(maximum + 1)

    ax.set_xlim(
        -limit,
        limit
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        "comparative_nll_change.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        "comparative_nll_change.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    create_radar_figure()
    create_output_rate_bar_figure()
    create_parseable_rate_bar_figure()
    create_nll_bar_figure()