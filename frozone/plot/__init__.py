def get_figure_args(nrow=1, ncol=1) -> dict[str]:
    return dict(
        figsize=(11 * ncol, 7 * nrow),
        fontsize=34,
        title_fontsize=0.4,
        legend_fontsize=1.1,
        axes_ticksize=1.1,
        other_rc_params={
            "lines.linewidth": 4,
        },
    )