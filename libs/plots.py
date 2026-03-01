""" Training and testing specific plots. """
# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
from typing import Union, Tuple, Iterable, Generator, Callable, List
from itertools import zip_longest, cycle
from pathlib import Path
from inspect import getsourcefile

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.axes import Axes as _Axes

# pylint: disable=no-member
import astropy.units as u
from astropy.table import Table

from uncertainties.unumpy import nominal_values, std_devs

from lightkurve import LightCurve as _LC, FoldedLightCurve as _FLC, LightCurveCollection as _LCC

from sed_fit.stellar_grids import StellarGrid
from sed_fit.fitter import model_func, iterate_theta

from .data.mist.read_mist_models import ISO


_this_dir = Path(getsourcefile(lambda:0)).parent

# Formatted equivalent of various param names for use in plot labels/captions
all_param_captions = {
    "rA_plus_rB":   r"$r_{\rm A}+r_{\rm B}$",
    "k":            r"$k$",
    "inc":          r"$i~[^{\circ}]$",
    "J":            r"$J$",
    "qphot":        r"$q_{phot}$",
    "ecosw":        r"$e\,\cos{\omega}$",
    "esinw":        r"$e\,\sin{\omega}$",
    "L3":           r"$L_{\rm 3}$",
    "bP":           r"$b_{\rm P}$",
    "ecc":          r"$e$",
    "e":            r"$e$",
    "rA":           r"$r_{\rm A}$",
    "rB":           r"$r_{\rm B}$",
    "light_ratio":  r"$L_{\rm B}/L_{\rm A}$",
    "LR":           r"$L_{\rm B}/L_{\rm A}$",
    "TeffA":        r"$T_{\rm eff,A}~[\text{K}]$",
    "TeffB":        r"$T_{\rm eff,B}~[\text{K}]$",
    "RA":           r"$R_{\rm A}~[\text{R}_{\odot}]$",
    "RB":           r"$R_{\rm B}~[\text{R}_{\odot}]$",
    "MA":           r"$M_{\rm A}~[\text{M}_{\odot}]$",
    "MB":           r"$M_{\rm B}~[\text{M}_{\odot}]$",
}


def plot_parameter_scatter(params: ArrayLike,
                           xdata: ArrayLike,
                           keys: List[str]=None,
                           cols: int=2,
                           suptitle: str=None,
                           ax_func: Callable[[str, _Axes], None]=None,
                           **format_kwargs) -> _Figure:
    """
    Plots a set of axes showing the scatter in each set of param values.

    :params: a structured array containing the set of params to plot
    :xdata: the shared x-axis data
    :keys: optional the list keys onto data within params to plot - if None all keys
    :cols: the number of columns in which to arrange the axes
    :suptitle: the optional overall figure title
    :ax_func: optional callback taking (key, ax) called on each Axes prior to applying format_kwargs
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    if keys is None:
        keys = list(params.dtype.names)

    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 1.5*rows),
                             sharex=True, constrained_layout=True)

    if suptitle is not None:
        fig.suptitle(suptitle)

    # Extract formatting for use with a single legend
    legend_loc, legend_ncol = None, 1
    if "legend_loc" in format_kwargs:
        legend_loc = format_kwargs.pop("legend_loc")
        legend_ncol = format_kwargs.pop("legend_ncol", 1)

    for ix, (ax, key) in enumerate(zip_longest(axes.flat, keys)):
        if ix < len(keys):
            ax.errorbar(xdata, nominal_values(params[key]), yerr=std_devs(params[key]),
                        fmt="o", c="tab:blue", capsize=None, fillstyle="none")

            ax.set_xticks(xdata, labels=xdata, minor=False)
            ax.set_ylabel(all_param_captions.get(key, key))

            if ax_func is not None:
                ax_func(key, ax)

            format_axes(ax, **format_kwargs)
        else:
            # Hide any unused axes
            ax.axis("off")

    if legend_loc is not None:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc=legend_loc, ncol=legend_ncol)

    return fig


def plot_lightcurves(lcs: Union[_LCC, _LC, _FLC],
                     column: str="flux",
                     ax_titles: Union[str, Iterable[str]]="LABEL",
                     normalize_lcs: bool=False,
                     cols: int=2,
                     ax_func: Callable[[int, _Axes, _LC], None]=None,
                     **format_kwargs) -> _Figure:
    """
    Creates a matplotlib figure and plots a grid of lightcurves on it, one per Axes.

    :lcs: the lightcurves to plot, one per matplotlib Axes
    :column: the lightcurve data column to plot of the y-axis
    :ax_titles: the titles to give each Axes or if a single str a meta key to read for each title
    :normalize_lcs: whether or not to normalize the y-axis data before plotting
    :cols: the number of columns on the grid of Axes
    :ax_func: callback taking (index, ax, LightCurve) called for each Axes/LC
    prior to applying format_kwargs
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
        ax_titles = [lc.meta.get(ax_titles, ax_titles) for lc in lcs]

    # Set up the figure and Axes
    rows = int(np.ceil(count_lcs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows),
                             sharey=True, constrained_layout=True)
    axes = [axes] if isinstance(axes, _Axes) else axes.flatten()

    for ix, (ax, lc, title) in enumerate(zip_longest(axes, lcs, ax_titles)):
        if ix < count_lcs:
            plot_lightcurve_on_axes(ax, lc, column, normalize_lcs)

            if ix == 0 and lc[column].unit == u.mag:
                ax.invert_yaxis()

            if ax_func is not None:
                ax_func(ix, ax, lc)

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


def plot_fitted_model_sed(sed: Table,
                          theta: ArrayLike,
                          model_grid: StellarGrid,
                          sed_flux_colname: str="sed_der_flux",
                          sed_flux_err_colname: str="sed_eflux",
                          sed_filter_colname: str="sed_filter",
                          sed_lambda_colname: str="sed_wl",
                          **format_kwargs) -> _Figure:
    """
    Wraps and extends plot_sed() so that the observed SED points are plotted plus the equivalent
    combined model SED data points from the fitted model. Additionally, the model SED points and
    full spectrum of each component star will be plotted.

    The data and axes will be coerced to units of x=wavelength [um] and y=nu*F(nu) [W / m^2].
    The axes will be set to log-log scale.

    :sed: the x-axis/wavelength datapoints
    :theta: the fitted parameters, as passed to sed_fit model_func, for which model fluxes are shown
    :model_grid: the StellarGrid from which the fitted fluxes are taken
    :sed_flux_colname: the name of the flux column in the sed table
    :sed_flux_err_colname: the name of the flux uncertainty column in the sed table
    :sed_filter_colname: the name of the filter column in the sed table
    :sed_lambda_colname: the name of the wavelength column in the sed table
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the final Figure
    """
    # Generate model SED fluxes at points x for each set of component star params in theta
    x = model_grid.get_filter_indices(sed[sed_filter_colname])
    theta_noms = nominal_values(theta)
    comp_fluxes = model_func(theta_noms, x, model_grid, combine=False) * model_grid.flux_unit

    # Need a set of plot formats/colours to cover reasonable number of components
    comp_fmts = ["*m", "+c", "xy", "2r"]
    comp_colors = ["m", "c", "y", "r"]

    # Plot the fitted model against the derredened SED + show each star's contribution
    nstars = comp_fluxes.shape[0]
    fig = plot_sed(sed[sed_lambda_colname].to(u.um),
                   [sed[sed_flux_colname].quantity, np.sum(comp_fluxes, axis=0)] +list(comp_fluxes),
                   [sed[sed_flux_err_colname].quantity, None] + [None]*nstars,
                   ["ob", ".k"] + list(_cycle_for(comp_fmts, nstars)),
                   ["dereddened SED", "fitted pair"] +[f"fitted star {i+1}" for i in range(nstars)],
                   **format_kwargs)

    # Plot the raw spectra for each component as a background
    spec_lams = model_grid.wavelengths * model_grid.wavelength_unit
    mask = spec_lams >= sed[sed_lambda_colname].quantity.min()
    mask &= spec_lams <= sed[sed_lambda_colname].quantity.max()
    for (teff, logg, rad, dist, av), c in zip(iterate_theta(theta_noms),
                                              _cycle_for(comp_colors, nstars)):
        spec_flux = model_grid.get_fluxes(teff, logg, 0, rad, dist, av) * model_grid.flux_unit
        fig.gca().plot(spec_lams[mask], spec_flux[mask], c=c, alpha=0.15, zorder=-100)
    return fig


def plot_mass_radius_diagram(masses: ArrayLike,
                             radii: ArrayLike,
                             labels: ArrayLike=None,
                             plot_zams: bool=False,
                                **format_kwargs) -> _Figure:
    """
    Plots a log(R) vs log(M) diagram with an optional ZAMS line.
    Returns the figure of the plot for the calling code to show or save.

    :masses: the mass values to plot on the x-axis in shape (#sets, #masses) or (#masses) for 1 set
    :radii: the radius values to plot on the y-axis in shape (#sets, #radii) or (#radii) for 1 set
    :labels: optional labels text for each set (if multiple sets) or item (if a single set)
    :plot_zams: whether or not to include a zero age main-sequence line on the figure
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the Figure
    """
    # Masses & radii both support multiple sets, but they must be the same shape
    if masses.shape != radii.shape:
        raise ValueError("masses and radii are not of the same shape")
    if labels is not None and len(labels) != masses.shape[0]:
        raise ValueError("labels do not match the masses or radii")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    labels = labels or [None] * masses.shape[0]
    for ix, (mass_vals, rad_vals, label) in enumerate(zip(masses, radii, labels)):
        ax.errorbar(x=nominal_values(mass_vals), xerr=std_devs(mass_vals),
                    y=nominal_values(rad_vals), yerr=std_devs(rad_vals),
                    fmt="o", ms=4, markeredgewidth=1, fillstyle="full", zorder=-ix, label=label)

    xlim = (min(0.1, max(ax.get_xlim()[0]*0.9, 1e-3)), max(20, ax.get_xlim()[1]*1.1))
    ylim = (min(0.1, max(ax.get_ylim()[0]*0.9, 1e-3)), max(20, ax.get_ylim()[1]*1.1))
    ax.set(xlabel= r"$\log{(M\,/\,{\rm M_{\odot}})}$", xscale="log", xlim=xlim,
           ylabel=r"$\log{(R\,/\,{\rm R_{\odot}})}$", yscale="log", ylim=ylim)

    xticks = [x for x in [-2, -1, 0, 1, 2, 3] if min(xlim) < 10**x < max(xlim)]
    ax.set_xticks([10**x for x in xticks], minor=False)
    ax.set_xticklabels(xticks, minor=False)

    yticks = [y for y in [-2, -1, 0, 1, 2, 3] if min(ylim) < 10**y < max(ylim)]
    ax.set_yticks([10**y for y in yticks], minor=False)
    ax.set_yticklabels(yticks, minor=False)

    if plot_zams:
        zams = _get_solar_isochrone_eep_values(eep=202, phase=0.0, cols=["star_mass", "log_R"])
        zmass = np.linspace(zams[0].min(), zams[0].max(), 250)
        zsort = np.argsort(zams[0])
        zrad = make_interp_spline(zams[0, zsort], zams[1, zsort], k=1)(zmass) # smoothing
        ax.plot(zmass, 10**zrad, ls="--", lw=1, c="k", zorder=-100, alpha=.5, label="ZAMS")

    format_axes(ax, **format_kwargs)
    return fig


def plot_hr_diagram(teffs: ArrayLike,
                    luminosities: ArrayLike,
                    labels: ArrayLike=None,
                    plot_zams: bool=False,
                    **format_kwargs) -> _Figure:
    """
    Plots a log(L) vs log(T_eff) Hertzsprung-Russell diagram with an optional ZAMS line.
    Returns the figure of the plot for the calling code to show or save.

    :teffs: the mass values to plot on the x-axis in shape (#sets, #teffs) or (#teffs) for 1 set
    :luminosities: the radius values to plot on the y-axis in shape (#sets, #lums) or (#lums) for 1 set
    :labels: optional labels text for each set (if multiple sets) or item (if a single set)
    :plot_zams: whether or not to include a zero age main-sequence line on the figure
    :format_kwargs: kwargs to be passed on to format_axes()
    :returns: the Figure
    """
    # Masses & radii both support multiple sets, but they must be the same shape
    if teffs.shape != luminosities.shape:
        raise ValueError("teffs and luminosities are not of the same shape")
    if labels is not None and len(labels) != teffs.shape[0]:
        raise ValueError("labels do not match the teffs or luminosities")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    labels = labels or [None] * teffs.shape[0]
    for ix, (teff_vals, lum_vals, label) in enumerate(zip(teffs, luminosities, labels)):
        ax.errorbar(x=nominal_values(teff_vals), xerr=std_devs(teff_vals),
                    y=nominal_values(lum_vals), yerr=std_devs(lum_vals),
                    fmt="o", ms=4, markeredgewidth=1, fillstyle="full", zorder=-ix, label=label)

    xlim = (min(3000, max(ax.get_xlim()[0]*0.9, 1e-3)), max(20000, ax.get_xlim()[1]*1.1))
    ylim = (min(0.1, max(ax.get_ylim()[0]*0.9, 1e-3)), max(20, ax.get_ylim()[1]*1.1))
    ax.set(xlabel=r"$\log{(T_{\rm eff}\,/\,{\rm K})}$", xscale="log", xlim=xlim,
           ylabel=r"$\log{(L\,/\,{\rm L_{\odot}})}$", yscale="log", ylim=ylim)

    xticks = [x for x in [3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8] if min(xlim)<10**x<max(xlim)]
    ax.set_xticks([10**x for x in xticks], minor=False)
    ax.set_xticklabels(xticks, minor=False)

    yticks = [y for y in [-3, -2, -1, 0, 1, 2, 3, 4, 5] if min(ylim)<10**y<max(ylim)]
    ax.set_yticks([10**y for y in yticks], minor=False)
    ax.set_yticklabels(yticks, minor=False)

    if plot_zams:
        zams = _get_solar_isochrone_eep_values(eep=202, phase=0.0, cols=["log_Teff", "log_L"])
        zteff = np.linspace(zams[0].min(), zams[0].max(), 250)
        zsort = np.argsort(zams[0])
        zlum = make_interp_spline(zams[0, zsort], zams[1, zsort], k=1)(zteff) # smoothing
        ax.plot(10**zteff, 10**zlum, ls="--", lw=1, c="k", zorder=-100, alpha=0.5, label="ZAMS")

    format_axes(ax, **format_kwargs)
    ax.tick_params(axis="x", which="minor", top=False, bottom=False, labelbottom=False)
    return fig


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

def _cycle_for(init_list: List, num_items: int):
    """
    Util func to cycle over the passed list and stop after yielding the required number of items
    """
    for i, v in enumerate(cycle(init_list)):
        if i == num_items:
            break
        yield v

def _get_solar_isochrone_eep_values(eep: int, phase: int, cols: List[str]) -> ArrayLike:
    """
    Gets the requested column values from the solar metallicity MIST isochrone,
    searching by eep and phase.

    Common eep values are 202 (ZAMS) & 453 (TAMS) with phase 0.0

    :iso: the MIST ISO to search
    :eep: the eep (equivalent evolutionary point) to find across the iso
    :phase: the phase to find across the iso, where 0.0 is main-sequence
    :cols: the columns to return the values for
    :returns: the requested data
    """
    iso_file = _this_dir / "data/mist/MIST_v1.2_vvcrit0.4_basic_isos" \
                        / "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso"
    iso = ISO(str(iso_file), verbose=False)

    rows = (ab[(ab["EEP"]==eep) & (ab["phase"]==phase)] for ab in iso.isos if eep in ab["EEP"])
    return np.array([list(row[0][cols]) for row in rows if len(row) > 0]).transpose()
