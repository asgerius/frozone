def get_figure_args(nrow=1, ncol=1, w=11, h=7, **kwargs) -> dict[str]:
    return dict(
        figsize=(w * ncol, h * nrow),
        fontsize=34,
        title_fontsize=0.4,
        legend_fontsize=1.1,
        axes_ticksize=1.1,
        other_rc_params={
            "lines.linewidth": 4,
        },
    ) | kwargs
