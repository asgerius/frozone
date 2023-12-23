def get_figure_args(nrow=1, ncol=1) -> dict[str]:
    return dict(
        figsize=(10 * ncol, 7 * nrow),
        fontsize=32,
        legend_fontsize=0.95,
        axes_ticksize=0.95
    )