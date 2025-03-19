""" Training and testing specific plots. """
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
from typing import Union, Tuple, Iterable, Generator, Callable
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

import astropy.units as u
from lightkurve import LightCurve as _LC, FoldedLightCurve as _FLC, LightCurveCollection as _LCC


def plot_lightcurves(lcs: Union[_LCC, _LC, _FLC],
                     column: str="flux",
                     ax_titles: Union[str, Iterable[str]]=None,
                     normalize_lcs: bool=False,
                     cols: int=2,
                     ax_func: Callable[[int, _Axes], None]=None,
                     **format_kwargs) -> _Figure:
    """
    Creates a matplotlib figure and plots a grid of lightcurves on it, one per Axes.

    :lcs: the lightcurves to plot, one per matplotlib Axes
    :column: the lightcurve data column to plot of the y-axis
    :ax_titles: the titles to give each Axes
    :normalize_lcs: whether or not to normalize the y-axis data before plotting
    :cols: the number of columns on the grid of Axes
    :ax_func: callback taking (ax index, ax) called for each Axes prior to applying format_kwargs
    :returns: the final Figure
    """
    # Ensure the data and titles are iterable, even if there is none
    if isinstance(lcs, (_LC, _FLC)):
        lcs = _LCC([lcs])
    count_lcs = len((list(lcs) if isinstance(lcs, Generator) else lcs))
    if ax_titles is None:
        ax_titles = []
    elif isinstance(ax_titles, str):
        ax_titles = [ax_titles] * count_lcs

    # Set up the figure and Axes
    rows = int(np.ceil(count_lcs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3*rows), sharey=True, constrained_layout=True)
    axes = [axes] if isinstance(axes, _Axes) else axes.flatten()

    for ix, (ax, lc, title) in enumerate(zip_longest(axes, lcs, ax_titles)):
        if ix < count_lcs:
            plot_lightcurve_on_axes(ax, lc, column, normalize_lcs)

            if ix == 0 and lc[column].unit == u.mag:
                ax.invert_yaxis()

            if ax_func is not None:
                ax_func(ix, ax)

            # Only want the y-label on the left most column as sharey is in play
            ax.set_ylabel(None if ix % cols else ax.get_ylabel())
            format_axes(ax, title=title, **format_kwargs)
        else:
            # Hide any unused axes
            ax.axis("off")

    return fig


def plot_lightcurve_on_axes(ax: _Axes, lc: Union[_LC, _FLC],
                            column: str="flux", normalize: bool=False):
    """
    Will plot the passed lightcurve on the passed axes with standardized formatting.

    :ax: the Axes
    :lc: the Lightcurve
    :column: the lightcurve data column to plot of the y-axis
    :normalize: whether or not to normalize the y-axis data before plotting
    """
    lc.scatter(ax=ax, column=column, s=2.0, marker=".", label=None, normalize=normalize)
    if lc[column].unit == u.mag:
        if column == "delta_mag":
            ax.set_ylabel("differential magnitude [mag]")


def format_axes(ax: _Axes, title: str=None,
                xlabel: str=None, ylabel: str=None,
                xticklable_top: bool=False, yticklabel_right: bool=False,
                invertx: bool=False, inverty: bool=False,
                xlim: Tuple[float, float]=None, ylim: Tuple[float, float]=None,
                minor_ticks: bool=True, legend_loc: str=None):
    """
    General purpose formatting function for a set of Axes. Will carry out the
    formatting instructions indicated by the arguments and will set all ticks
    to internal and on all axes.

    :ax: the Axes to format
    :title: optional title to give the axes, overriding prior code - set to "" to surpress
    :xlabel: optional x-axis label text to set, overriding prior code - set to "" to surpress
    :ylabel: optional y-axis label text to set, overriding prior code - set to "" to surpress
    :xticklabel_top: move the x-axis ticklabels and label to the top
    :yticklabel_right: move the y-axis ticklabels and label to the right
    :invertx: invert the x-axis
    :inverty: invert the y-axis
    :xlim: set the lower and upper limits on the x-axis
    :ylim: set the lower and upper limits on the y-axis
    :minor_ticks: enable or disable minor ticks on both axes
    :legend_loc: if set will enable to legend and set its position.
    For available values see matplotlib legend(loc="")
    """
    # pylint: disable=too-many-arguments
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if invertx:
        ax.invert_xaxis()
    if inverty:
        ax.invert_yaxis()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if minor_ticks:
        ax.minorticks_on()
    else:
        ax.minorticks_off()
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, bottom=True, left=True, right=True)
    if xticklable_top:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    if yticklabel_right:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    if legend_loc:
        ax.legend(loc=legend_loc)
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()
