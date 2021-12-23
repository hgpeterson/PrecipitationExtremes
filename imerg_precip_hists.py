################################################################################
# Analyzing IMERG Precipitation Statistics Using Histograms
################################################################################

import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import colors
import warnings
warnings.filterwarnings('ignore')
import cmip6_lib

################################################################################
# functions
################################################################################

def get_precip_pdf(dfiles, nbins, latslice, lonslice):
    """
        precip, hist, bins = get_precip_pdf(dfiles, nbins, latslice, lonslice)

    Get the precipitation pdf from IMERG data.
    """
    # mfdataset for multiple files
    ds = xr.open_mfdataset(dfiles)
    precip = ds["HQprecipitation"] # mm day-1

    # time and space slice
    if latslice == "extrop":
        precip = precip.sel(lon=lonslice).where(np.abs(precip.lat) >= 30, drop=True)
    else:
        precip = precip.sel(lat=latslice, lon=lonslice)

    # weight by cos(ϕ)
    weights = da.ones_like(precip)
    cosϕ = np.cos(np.deg2rad(precip.lat))
    weights *= np.tile(cosϕ, (len(precip.lon), 1))
    precip *= weights

    # setup bins
    pmin = 10**-5
    pmax = 10**3
    bins = np.zeros(nbins)
    bins[1:] = 10**np.linspace(np.log10(pmin), np.log10(pmax), nbins-1) # include 0 as a bin

    # compute pdf
    hist, bins = da.histogram(precip, bins=bins, density=False)    

    # actually compute
    hist = hist.compute()

    return precip, hist, bins

def get_hists():
    """
        get_hists()
        
    Save .npz files of histograms of precipitation for 2000-2005 and 2016-2021
    """
    # compute histograms for each month
    for year in range(2000, 2006):
        for month in range(1, 13):
            if year == 2000 and month < 6:
                continue
            if year == 2005 and month > 5:
                continue
            year = f"{year}"
            month = f"{month:02d}"
            dfiles = path + "3B-DAY.MS.MRG.3IMERG." + year + month + "*.nc4"

            precip, hist, bins = get_precip_pdf(dfiles, nbins, latslice, lonslice)
            np.savez(f"data/imerg{domain}{year}{month}.npz", hist=hist, bins=bins)
            print(f"data/imerg{domain}{year}{month}.npz")
    for year in range(2016, 2022):
        for month in range(1, 13):
            if year == 2016 and month < 6:
                continue
            if year == 2021 and month > 5:
                continue
            year = f"{year}"
            month = f"{month:02d}"
            dfiles = path + "3B-DAY.MS.MRG.3IMERG." + year + month + "*.nc4"

            precip, hist, bins = get_precip_pdf(dfiles, nbins, latslice, lonslice)
            np.savez(f"data/imerg{domain}{year}{month}.npz", hist=hist, bins=bins)
            print(f"data/imerg{domain}{year}{month}.npz")

    # add histograms together for 2000-2005 and 2016-2021
    data = np.load(f"data/imerg{domain}200006.npz")
    bins = data["bins"]
    hist_past    = np.zeros(len(bins)-1)
    hist_present = np.zeros(len(bins)-1)
    for year in range(2000, 2006):
        for month in range(1, 13):
            if year == 2000 and month < 6:
                continue
            if year == 2005 and month > 5:
                continue
            data = np.load(f"data/imerg{domain}{year}{month:02d}.npz")
            hist_past += data["hist"]
    for year in range(2016, 2022):
        for month in range(1, 13):
            if year == 2016 and month < 6:
                continue
            if year == 2021 and month > 5:
                continue
            data = np.load(f"data/imerg{domain}{year}{month:02d}.npz")
            hist_present += data["hist"]
    np.savez("data/imerg2000-2005.npz", hist=hist_past, bins=bins)
    np.savez("data/imerg2016-2021.npz", hist=hist_present, bins=bins)

def load_data(file):
    """
        bins, hist, k, θ = load_data(file)

    Load histogram data from .npz file.
    """
    # load
    data = np.load(file)
    bins = data["bins"]
    hist = data["hist"]

    # weight
    bins_widths, hist = cmip6_lib.hist_to_pdf(bins, hist)

    # compute gamma fit
    d = np.where(np.logical_and(bins > fit_min, bins < fit_max))
    k, θ = cmip6_lib.gamma_fit_from_hist(bins[d], hist[d][1:])

    return bins, hist, k, θ

def plot_hists():
    """
        plot_hists()
        
    Plot precip histograms for past/present. 
    """
    bins_c, hist_c, k_c, θ_c = load_data("data/imerg2000-2005.npz")
    x_c = 10**np.linspace(-6, np.log(np.max(bins_c)), 1000)
    pdf_c = cmip6_lib.gamma_dist(x_c, k_c, θ_c)

    bins_w, hist_w, k_w, θ_w = load_data("data/imerg2016-2021.npz")
    x_w = 10**np.linspace(-6, np.log(np.max(bins_w)), 1000)
    pdf_w = cmip6_lib.gamma_dist(x_w, k_w, θ_w)

    fig, ax = plt.subplots()
    ax.loglog(bins_c[:-1], hist_c, c="tab:blue", label="2000--2005")
    # ax.loglog(x_c, pdf_c, c="tab:blue", ls="--", label="2000--2005 Gamma Fit")
    ax.loglog(bins_w[:-1], hist_w, c="tab:orange", label="2016--2021")
    # ax.loglog(x_w, pdf_w, c="tab:orange", ls="--", label="2016--2021 Gamma Fit")

    # compare with GFDL
    d = np.load("data/hists_pr_day_GFDL-CM4_ssp585_r1i1p1f1_gr1.npz")
    bins_c, hist_c = d["bins_c"], d["hist_c"]
    bins_w, hist_w = d["bins_w"], d["hist_w"]
    bins_widths, hist_c = cmip6_lib.hist_to_pdf(bins_c, hist_c)
    bins_widths, hist_w = cmip6_lib.hist_to_pdf(bins_w, hist_w)
    ax.loglog(bins_c[:-1], hist_c, c="tab:blue",   ls=":", label="GFDL CM4 Control")
    ax.loglog(bins_w[:-1], hist_w, c="tab:orange", ls=":", label="GFDL CM4 Warming")

    ax.legend()
    ax.set_xlabel("Precipitation (mm day$^{-1}$)")
    ax.set_ylabel("Probability")
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e-13, 1e5])
    cmip6_lib.savefig("imerg_hists_zoom.png")
    ax.axvline(fit_min, lw=1.0, c="k", ls=":", alpha=0.5)
    ax.axvline(fit_max, lw=1.0, c="k", ls=":", alpha=0.5)
    ax.set_xlim([1e-5, 1e3])
    ax.set_ylim([1e-13, 1e15])
    cmip6_lib.savefig("imerg_hists.png")
    plt.close()

def plot_pctls():
    """
        plot_pctls()
        
    Make percentile plot for imerg data.
    """
    bins_c, hist_c, k_c, θ_c = load_data("data/imerg2000-2005.npz")
    bins_w, hist_w, k_w, θ_w = load_data("data/imerg2016-2021.npz")

    i_bin_min = np.argmin(np.abs(bins_c - fit_min))
    nbins = len(bins_c)
    b_c = np.zeros(nbins - i_bin_min + 1)
    b_c[1:] = bins_c[i_bin_min:]
    h_c = np.zeros(nbins - i_bin_min)
    h_c[0] = hist_c[np.where(bins_c < 1e-1)].sum() 
    h_c[1:] = hist_c[i_bin_min:]

    i_bin_min = np.argmin(np.abs(bins_w - fit_min))
    nbins = len(bins_w)
    b_w = np.zeros(nbins - i_bin_min + 1)
    b_w[1:] = bins_w[i_bin_min:]
    h_w = np.zeros(nbins - i_bin_min)
    h_w[0] = hist_w[np.where(bins_w < 1e-1)].sum() 
    h_w[1:] = hist_w[i_bin_min:]

    # fit_min = 1e-2
    # i_bin_min = np.argmin(np.abs(bins_c - fit_min))
    # b_c = bins_c[i_bin_min:]
    # h_c = hist_c[i_bin_min:]

    # i_bin_min = np.argmin(np.abs(bins_w - fit_min))
    # b_w = bins_w[i_bin_min:]
    # h_w = hist_w[i_bin_min:]

    # d = np.where(np.logical_and(bins_c > 1e-1, bins_c < 2e2))
    # bins_c = bins_c[d]
    # hist_c = hist_c[d][1:]
    # d = np.where(np.logical_and(bins_w > 1e-1, bins_w < 2e2))
    # bins_w = bins_w[d]
    # hist_w = hist_w[d][1:]

    # bins_c = bins_c[1:]
    # hist_c = hist_c[1:]
    # bins_w = bins_w[1:]
    # hist_w = hist_w[1:]

    # bin_widths, pdf = cmip6_lib.hist_to_pdf(bins_c, hist_c)
    # pctls = np.cumsum(pdf*bin_widths)

    fig, ax = plt.subplots()
    ΔT = 1 # for now have ratios be in %, not % K-1
    # axtwin = cmip6_lib.pctl_plot_hist(ax, ΔT, bins_c, hist_c, k_c, θ_c, bins_w, hist_w, k_w, θ_w)
    axtwin = cmip6_lib.pctl_plot_hist(ax, ΔT, b_c, h_c, k_c, θ_c, b_w, h_w, k_w, θ_w)
    xticklabels = ["9", "90", "99", "99.9", "99.99"]
    xticks = np.array([0.09, 0.9, 0.99, 0.999, 0.9999])
    ax.set_xlim([0, 13])
    ax.set_xticks(cmip6_lib.x_transform(xticks))
    ax.set_xticklabels(xticklabels)
    axtwin.set_ylim([-20, 60])
    axtwin.set_ylabel("Change (\%)", c="tab:green")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Precipitation (mm day$^{-1}$)")
    cmip6_lib.savefig("imerg_pctls.png")
    plt.close()


################################################################################
# run
################################################################################

# bins
nbins = 1000

# min/max precip (mm day-1) for computing Gamma fit
fit_min = 1e-1
fit_max = 2e2

# lat slice
latslice = slice(-90, 90)
# latslice = slice(-30, 30)
# latslice = "extrop"

# lon slice
lonslice = slice(0, 360)

# domain
if latslice == slice(-90, 90):
    domain = ""
elif latslice == slice(-30, 30):
    domain = "trop"
elif latslice == "extrop":
    domain = "extrop"

# path
path = "/export/data1/hgpeterson/imerg/"

# get hists
# get_hists()

# plot hists
plot_hists()

# plot pctls
plot_pctls()