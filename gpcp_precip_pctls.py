################################################################################
# Analyzing GPCP Precipitation Statistics Using Percentiles
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

def get_precip_pctls(dfiles, pctls, timeslice, latslice, lonslice):
    """
        precip, precip_pctls, precip_pctls_boot = get_precip_pctls(dfiles, pctls, timeslice, latslice, lonslice)

    Get the precipitation percentiles from monthly GCPC data.
    """
    # mfdataset for multiple files
    ds = xr.open_mfdataset(dfiles)
    precip = ds["precip"] # mm day-1

    # time and space slice
    if latslice == "extrop":
        precip = precip.sel(time=timeslice, longitude=lonslice).where(np.abs(precip.latitude) >= 30, drop=True)
    else:
        precip = precip.sel(time=timeslice, latitude=latslice, longitude=lonslice)

    # weight by cos(ϕ)
    weights = da.ones_like(precip)
    cosϕ = np.cos(np.deg2rad(precip.latitude))
    weights *= np.tile(cosϕ, (len(precip.longitude), 1))
    precip *= weights

    # load
    precip = precip.values
    
    # compute pctls
    precip_pctls = np.percentile(precip, 100*pctls, interpolation="linear")
    
    # bootstrap indices
    datasize = precip.shape[0]
    samplesize = datasize
    blocksize = 10 # lagrangian decorrelation time ~ 10 days
    nboot = 200
    bootstrap_indices = cmip6_lib.stationary_bootstrap(datasize, samplesize, blocksize, nboot)
    
    # compute pctls of bootstrapped data
    npctls = pctls.shape[0]
    precip_pctls_boot = np.zeros((nboot, npctls))
    for i in range(nboot):
        precip_pctls_boot[i, :] = np.percentile(precip[bootstrap_indices[i, :], :, :], 100*pctls, interpolation="linear")
    
    return precip, precip_pctls, precip_pctls_boot

"""
    precip_c, precip_pctls_c, precip_pctls_boot_c, precip_w, precip_pctls_w, precip_pctls_boot_w, 
    μ_c, σ_c, μ_w, σ_w = get_data_gpcp(dfiles, pctls, latslice, lonslice)
"""
def get_data_gpcp(dfiles, pctls, latslice, lonslice):
    # control: first 21 years
    print("computing control pctls.")
    timeslice = slice("1979", "2000")
    precip_c, precip_pctls_c, precip_pctls_boot_c = get_precip_pctls(dfiles, pctls, timeslice, latslice, lonslice)

    # warming: last 20 years
    print("computing warming pctls")
    timeslice = slice("2000", "2020")
    precip_w, precip_pctls_w, precip_pctls_boot_w = get_precip_pctls(dfiles, pctls, timeslice, latslice, lonslice)
    
    # mean and std
    print("computing mean and std")
    μ_c, σ_c = cmip6_lib.get_mean_std(precip_c)
    μ_w, σ_w = cmip6_lib.get_mean_std(precip_w)
    
    return precip_c, precip_pctls_c, precip_pctls_boot_c, precip_w, precip_pctls_w, precip_pctls_boot_w, μ_c, σ_c, μ_w, σ_w

def save_data():
    """
        get_hists()
      
    Save .npz files of pctls of precipitation for 1979-2000 and 2000-2020.
    """
    precip_c, precip_pctls_c, precip_pctls_boot_c, precip_w, precip_pctls_w, precip_pctls_boot_w, μ_c, σ_c, μ_w, σ_w = get_data_gpcp(dfiles, pctls, latslice, lonslice)
    np.savez("data/gpcp1979-2000.npz", precip=precip_c, precip_pctls=precip_pctls_c, precip_pctls_boot=precip_pctls_boot_c, μ=μ_c, σ=σ_c)
    np.savez("data/gpcp2000-2020.npz", precip=precip_w, precip_pctls=precip_pctls_w, precip_pctls_boot=precip_pctls_boot_w, μ=μ_w, σ=σ_w)

# def load_data(file):
#     """
#         bins, hist, k, θ = load_data(file)

#     Load histogram data from .npz file.
#     """
#     # load
#     data = np.load(file)
#     bins = data["bins"]
#     hist = data["hist"]

#     # weight
#     bins_widths, hist = cmip6_lib.hist_to_pdf(bins, hist)

#     # compute gamma fit
#     d = np.where(np.logical_and(bins > fit_min, bins < fit_max))
#     k, θ = cmip6_lib.gamma_fit_from_hist(bins[d], hist[d][1:])

#     return bins, hist, k, θ

# def plot_hists():
#     """
#         plot_hists()
        
#     Plot precip histograms for past/present. 
#     """
#     bins_c, hist_c, k_c, θ_c = load_data("data/imerg2000-2005.npz")
#     x_c = 10**np.linspace(-6, np.log(np.max(bins_c)), 1000)
#     pdf_c = cmip6_lib.gamma_dist(x_c, k_c, θ_c)

#     bins_w, hist_w, k_w, θ_w = load_data("data/imerg2016-2021.npz")
#     x_w = 10**np.linspace(-6, np.log(np.max(bins_w)), 1000)
#     pdf_w = cmip6_lib.gamma_dist(x_w, k_w, θ_w)

#     fig, ax = plt.subplots()
#     ax.loglog(bins_c[:-1], hist_c, c="tab:blue", label="2000--2005")
#     # ax.loglog(x_c, pdf_c, c="tab:blue", ls="--", label="2000--2005 Gamma Fit")
#     ax.loglog(bins_w[:-1], hist_w, c="tab:orange", label="2016--2021")
#     # ax.loglog(x_w, pdf_w, c="tab:orange", ls="--", label="2016--2021 Gamma Fit")

#     # compare with GFDL
#     d = np.load("data/hists_pr_day_GFDL-CM4_ssp585_r1i1p1f1_gr1.npz")
#     bins_c, hist_c = d["bins_c"], d["hist_c"]
#     bins_w, hist_w = d["bins_w"], d["hist_w"]
#     bins_widths, hist_c = cmip6_lib.hist_to_pdf(bins_c, hist_c)
#     bins_widths, hist_w = cmip6_lib.hist_to_pdf(bins_w, hist_w)
#     ax.loglog(bins_c[:-1], hist_c, c="tab:blue",   ls=":", label="GFDL CM4 Control")
#     ax.loglog(bins_w[:-1], hist_w, c="tab:orange", ls=":", label="GFDL CM4 Warming")

#     ax.legend()
#     ax.set_xlabel("Precipitation (mm day$^{-1}$)")
#     ax.set_ylabel("Probability")
#     ax.set_xlim([1e-1, 1e3])
#     ax.set_ylim([1e-13, 1e5])
#     cmip6_lib.savefig("imerg_hists_zoom.png")
#     ax.axvline(fit_min, lw=1.0, c="k", ls=":", alpha=0.5)
#     ax.axvline(fit_max, lw=1.0, c="k", ls=":", alpha=0.5)
#     ax.set_xlim([1e-5, 1e3])
#     ax.set_ylim([1e-13, 1e15])
#     cmip6_lib.savefig("imerg_hists.png")
#     plt.close()

# def plot_pctls():
#     """
#         plot_pctls()
        
#     Make percentile plot for imerg data.
#     """
#     bins_c, hist_c, k_c, θ_c = load_data("data/imerg2000-2005.npz")
#     bins_w, hist_w, k_w, θ_w = load_data("data/imerg2016-2021.npz")

#     i_bin_min = np.argmin(np.abs(bins_c - fit_min))
#     nbins = len(bins_c)
#     b_c = np.zeros(nbins - i_bin_min + 1)
#     b_c[1:] = bins_c[i_bin_min:]
#     h_c = np.zeros(nbins - i_bin_min)
#     h_c[0] = hist_c[np.where(bins_c < 1e-1)].sum() 
#     h_c[1:] = hist_c[i_bin_min:]

#     i_bin_min = np.argmin(np.abs(bins_w - fit_min))
#     nbins = len(bins_w)
#     b_w = np.zeros(nbins - i_bin_min + 1)
#     b_w[1:] = bins_w[i_bin_min:]
#     h_w = np.zeros(nbins - i_bin_min)
#     h_w[0] = hist_w[np.where(bins_w < 1e-1)].sum() 
#     h_w[1:] = hist_w[i_bin_min:]

#     # fit_min = 1e-2
#     # i_bin_min = np.argmin(np.abs(bins_c - fit_min))
#     # b_c = bins_c[i_bin_min:]
#     # h_c = hist_c[i_bin_min:]

#     # i_bin_min = np.argmin(np.abs(bins_w - fit_min))
#     # b_w = bins_w[i_bin_min:]
#     # h_w = hist_w[i_bin_min:]

#     # d = np.where(np.logical_and(bins_c > 1e-1, bins_c < 2e2))
#     # bins_c = bins_c[d]
#     # hist_c = hist_c[d][1:]
#     # d = np.where(np.logical_and(bins_w > 1e-1, bins_w < 2e2))
#     # bins_w = bins_w[d]
#     # hist_w = hist_w[d][1:]

#     # bins_c = bins_c[1:]
#     # hist_c = hist_c[1:]
#     # bins_w = bins_w[1:]
#     # hist_w = hist_w[1:]

#     # bin_widths, pdf = cmip6_lib.hist_to_pdf(bins_c, hist_c)
#     # pctls = np.cumsum(pdf*bin_widths)

#     fig, ax = plt.subplots()
#     ΔT = 1 # for now have ratios be in %, not % K-1
#     # axtwin = cmip6_lib.pctl_plot_hist(ax, ΔT, bins_c, hist_c, k_c, θ_c, bins_w, hist_w, k_w, θ_w)
#     axtwin = cmip6_lib.pctl_plot_hist(ax, ΔT, b_c, h_c, k_c, θ_c, b_w, h_w, k_w, θ_w)
#     xticklabels = ["9", "90", "99", "99.9", "99.99"]
#     xticks = np.array([0.09, 0.9, 0.99, 0.999, 0.9999])
#     ax.set_xlim([0, 13])
#     ax.set_xticks(cmip6_lib.x_transform(xticks))
#     ax.set_xticklabels(xticklabels)
#     axtwin.set_ylim([-20, 60])
#     axtwin.set_ylabel("Change (\%)", c="tab:green")
#     ax.set_xlabel("Percentile")
#     ax.set_ylabel("Precipitation (mm day$^{-1}$)")
#     cmip6_lib.savefig("imerg_pctls.png")
#     plt.close()


# ################################################################################
# # run
# ################################################################################

# # bins
# nbins = 1000

# # min/max precip (mm day-1) for computing Gamma fit
# fit_min = 1e-1
# fit_max = 2e2

# # lat slice
# latslice = slice(-90, 90)
# # latslice = slice(-30, 30)
# # latslice = "extrop"

# # lon slice
# lonslice = slice(0, 360)

# # domain
# if latslice == slice(-90, 90):
#     domain = ""
# elif latslice == slice(-30, 30):
#     domain = "trop"
# elif latslice == "extrop":
#     domain = "extrop"

# # path
# path = "/export/data1/hgpeterson/imerg/"

# # get hists
# # get_hists()

# # plot hists
# plot_hists()

# # plot pctls
# plot_pctls()