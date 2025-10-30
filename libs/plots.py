""" Training and testing specific plots. """
# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
from typing import Union, Tuple, Iterable, Generator, Callable, List
from itertools import zip_longest

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table

from uncertainties.unumpy import nominal_values

from lightkurve import LightCurve as _LC, FoldedLightCurve as _FLC, LightCurveCollection as _LCC

from sed_fit.stellar_grids import StellarGrid
from sed_fit.sed_fit import model_func

def plot_sed(x: u.Quantity,
             fluxes: List[u.Quantity],
             flux_errs: List[u.Quantity]=None,
             fmts: List[str]=None,
             labels: List[str]=None,
             figsize: Tuple[float, float]=(6, 4),
             **format_kwargs) -> _Figure:
    """
    Will create a new figure with a single set of axes and will plot one or more sets of SED flux
    datapoints.

    The data and axes will be coerced to units of x=wavelength [um] and y=nu*F(nu) [W / m^2].
    The axes will be set to log-log scale.

    :x: the x-axis/wavelength datapoints
    :fluxes: one or more sets of flux values at frequencies/wavelengths x
    :flux_errs: optional corresponding flux error bars
    :fmts: fmt options for each set of fluxes or leave as None for default (see the matplotlib docs
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot)
    :labels: optional labels for the fluxes
    :title: optional title for the plot
    :figsize: optional size for the figure
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    if isinstance(fluxes, u.Quantity):
        fluxes = [fluxes]
    if isinstance(flux_errs, u.Quantity):
        flux_errs = [flux_errs]
    if isinstance(fmts, str):
        fmts = [fmts]
    if isinstance(labels, str):
        labels = [labels] + [None] * len(fluxes)-1

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    vfv_unit = u.W / u.m**2
    lam = x.to(u.um, equivalencies=u.spectral())
    freq = x.to(u.Hz, equivalencies=u.spectral())

    for flux, flux_err, fmt, label in zip(fluxes, flux_errs, fmts, labels):
        vfv, vfv_err = None, None
        if flux is not None:
            if flux.unit.is_equivalent(vfv_unit):
                vfv = flux.to(vfv_unit).to(vfv_unit , equivalencies=u.spectral_density(freq))
            else:
                vfv = (flux * freq).to(vfv_unit , equivalencies=u.spectral_density(freq))
        if flux_err is not None:
            if flux_err.unit.is_equivalent(vfv_unit):
                vfv_err = flux_err.to(vfv_unit , equivalencies=u.spectral_density(freq))
            else:
                vfv_err = (freq * flux_err).to(vfv_unit, equivalencies=u.spectral_density(freq))

        if vfv is not None:
            ax.errorbar(lam, vfv, vfv_err, fmt=fmt, alpha=0.5, label=label)

    ax.set(xscale="log", xlabel=f"Wavelength [{u.um:latex_inline}]",
           yscale="log", ylabel=f"${{\\rm \\nu F(\\nu)}}$ [{u.W/u.m**2:latex_inline}]")
    ax.grid(True, which="both", axis="both", alpha=0.33, color="lightgray")
    legend_loc = "best" if labels is not None and any(l is not None for l in labels) else None
    format_axes(ax, legend_loc=legend_loc, **format_kwargs)
    return fig


def plot_fitted_model(sed: Table,
                      theta: ArrayLike,
                      model_grid: StellarGrid,
                      sed_flux_colname: str="sed_der_flux",
                      sed_flux_err_colname: str="sed_eflux",
                      sed_filter_colname: str="sed_filter",
                      sed_lambda_colname: str="sed_wl",
                      **format_kwargs):
    """
    Wraps and extends plot_sed() so that the observed SED points are plotted plus the equivalent
    combined model SED data points from the fitted model. Additionally, the model SED points and
    full spectrum of each component star will be plotted.

    The data and axes will be coerced to units of x=wavelength [um] and y=nu*F(nu) [W / m^2].
    The axes will be set to log-log scale.

    :sed: the x-axis/wavelength datapoints
    :theta: the fitting parameters as passed to sed_fit model_func
    :model_grid: the StellarGrid supplying the model fluxes to the fitting
    :sed_flux_colname:
    :sed_flux_err_colname:
    :sed_filter_colname:
    :sed_lambda_colname:
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    # Generate model SED fluxes at points x for each set of component star params in theta
    x = model_grid.get_filter_indices(sed[sed_filter_colname])
    th = nominal_values(theta)
    comp_fluxes = model_func(th, x, model_grid, combine=False) * model_grid.flux_unit
    sys_flux = np.sum(comp_fluxes, axis=0)

    # Plot the fitted model against the derredened SED + show each star's contribution
    fluxes = [sed[sed_flux_colname].quantity, sys_flux, comp_fluxes[0], comp_fluxes[1]]
    flux_errs = [sed[sed_flux_err_colname].quantity, None, None, None]
    fig = plot_sed(x=sed[sed_lambda_colname].to(u.um), fluxes=fluxes, flux_errs=flux_errs,
                   fmts=["ob", ".k", "*g", "xr"],
                   labels=["dereddened SED", "fitted pair", "fitted star A", "fitted star B"],
                   **format_kwargs)

    # Plot the raw spectra for each component as a background
    spec_lams = model_grid.wavelengths * model_grid.wavelength_unit
    mask = spec_lams >= sed[sed_lambda_colname].quantity.min()
    mask &= spec_lams <= sed[sed_lambda_colname].quantity.max()
    dist = th[-2]
    for teff, rad, logg, c in [(th[0], th[2], th[4], "g"), (th[1], th[3], th[5], "r")]:
        spec_flux = model_grid.get_fluxes(teff, logg, 0, rad, dist) * model_grid.flux_unit
        fig.gca().plot(spec_lams[mask], spec_flux[mask], c=c, alpha=0.15, zorder=-100)
    return fig


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
    :format_kwargs: kwargs to be passed on to format_axes()
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
