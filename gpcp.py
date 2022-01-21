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

def get_precip(dfiles, timeslice, latslice, lonslice):
    """
        precip = get_precip(dfiles, timeslice, latslice, lonslice) 

    Load GPCP precipitation data.
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
    weights *= np.tile(cosϕ, (len(precip.longitude), 1)).T
    precip *= weights

    return precip

def get_pctls(dfiles, pctls, timeslice, latslice, lonslice):
    """
        precip, precip_pctls, precip_pctls_boot = get_pctls(dfiles, pctls, timeslice, latslice, lonslice)

    Get the precipitation percentiles from monthly GCPC data.
    """
    # load
    precip = get_precip(dfiles, timeslice, latslice, lonslice)
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

def get_hist(dfiles, nbins, timeslice, latslice, lonslice):
    """
        hist, bins = get_hist(dfiles, nbins, timeslice, latslice, lonslice)

    Get the precipitation histogram.
    """
    # load 
    precip = get_precip(dfiles, timeslice, latslice, lonslice)

    # setup bins
    pmin = 10**-5
    pmax = 10**3
    bins = np.zeros(nbins)
    bins[1:] = 10**np.linspace(np.log10(pmin), np.log10(pmax), nbins-1) # include 0 as a bin

    # compute pdf
    hist, bins = da.histogram(precip, bins=bins, density=False)    

    # actually compute
    hist = hist.compute()

    return hist, bins

"""
    get_data(dfiles, pctls, nbins, latslice, lonslice)

Save pctls data fro gpcp for 1979-2000 and 2000-2020.
"""
def get_data(dfiles, pctls, nbins, latslice, lonslice):
    # control: first 21 years
    print("computing control pctls.")
    timeslice = slice("1979", "2000")
    print("\tpercentiles")
    precip_c, precip_pctls_c, precip_pctls_boot_c = get_pctls(dfiles, pctls, timeslice, latslice, lonslice)
    print("\thist")
    hist_c, bins_c = get_hist(dfiles, nbins, timeslice, latslice, lonslice)

    # warming: last 20 years
    print("computing warming pctls")
    timeslice = slice("2000", "2020")
    print("\tpercentiles")
    precip_w, precip_pctls_w, precip_pctls_boot_w = get_pctls(dfiles, pctls, timeslice, latslice, lonslice)
    print("\thist")
    hist_w, bins_w = get_hist(dfiles, nbins, timeslice, latslice, lonslice)
    
    # mean and std
    print("computing mean and std")
    μ_c, σ_c = cmip6_lib.get_mean_std(precip_c)
    μ_w, σ_w = cmip6_lib.get_mean_std(precip_w)

    np.savez(f"data/gpcp_{domain}_{freq}_1979-2000.npz", pctls=pctls, precip=precip_c, 
             precip_pctls=precip_pctls_c, precip_pctls_boot=precip_pctls_boot_c, 
             bins=bins_c, hist=hist_c,
             μ=μ_c, σ=σ_c)
    np.savez(f"data/gpcp_{domain}_{freq}_2000-2020.npz", pctls=pctls, precip=precip_w, 
             precip_pctls=precip_pctls_w, precip_pctls_boot=precip_pctls_boot_w, 
             bins=bins_w, hist=hist_w,
             μ=μ_w, σ=σ_w)

def load_data(file):
    """
        pctls, precip, precip_pctls, precip_pctls_boot, bins, hist, μ, σ, k, θ = load_data(file)

    Load pctl data from .npz file.
    """
    # load
    data = np.load(file)
    pctls = data["pctls"]
    precip = data["precip"]
    precip_pctls = data["precip_pctls"]
    precip_pctls_boot = data["precip_pctls_boot"]
    bins = data["bins"]
    hist = data["hist"]
    μ = data["μ"]
    σ = data["σ"]

    # compute gamma fit
    k, θ = cmip6_lib.gamma_moments(μ, σ)

    return pctls, precip, precip_pctls, precip_pctls_boot, bins, hist, μ, σ, k, θ

def plot_pctls():
    fig, ax = plt.subplots()

    pctls, precip_c, precip_pctls_c, precip_pctls_boot_c, bins_c, hist_c, μ_c, σ_c, k_c, θ_c = load_data(f"data/gpcp_{domain}_{freq}_1979-2000.npz")
    pctls, precip_w, precip_pctls_w, precip_pctls_boot_w, bins_w, hist_w, μ_w, σ_w, k_w, θ_w = load_data(f"data/gpcp_{domain}_{freq}_2000-2020.npz")

    # means and standard deviations from pdfs
    mean_scaling = μ_w/μ_c - 1

    # warming fit based on theory
    if freq == "monthly":
        i99 = np.argmin(np.abs(pctls - 0.99))
        extreme_scaling = precip_pctls_w[i99]/precip_pctls_c[i99] - 1
    else:
        extreme_scaling = precip_pctls_w[-1]/precip_pctls_c[-1] - 1
    θ_w_theory = θ_c*(1 + extreme_scaling)
    k_w_theory = k_c*(1 + mean_scaling - extreme_scaling)

    ΔT_global = 1 # no dT for now
    axtwin = cmip6_lib.pctl_plot(ax, ΔT_global, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w_theory, θ_w_theory)

    # label plot
    ax.annotate(f"GPCP {freq.capitalize()}", (0.08, 0.8), xycoords="axes fraction")
    
    # temp change 
    # ax.annotate("$\Delta T = {:1.2f}$ K".format(ΔT), (0.08, 0.8), xycoords="axes fraction")
    
    # axtwin.set_ylabel("Change (\% K$^{-1}$)", c="tab:green")
    axtwin.set_ylabel("Change (\%)", c="tab:green")
    ax.set_ylabel("Precipitation (mm day$^{-1}$)")
    ax.set_xlabel("Percentile")
        
    custom_lines = [Line2D([0], [0], ls="-", c="tab:blue"),
                   Line2D([0], [0], ls="-", c="tab:orange"),
                   Line2D([0], [0], ls="-", c="tab:green"),
                   Line2D([0], [0], ls="--", c="k")]
    custom_handles = ["1979--2000", "2000--2020", "ratio", "gamma fit"]
    ax.legend(custom_lines, custom_handles)

    plt.savefig(f"gpcp_{domain}_{freq}_pctl.png")
    print(f"gpcp_{domain}_{freq}_pctl.png")
    plt.savefig(f"gpcp_{domain}_{freq}_pctl.pdf")
    print(f"gpcp_{domain}_{freq}_pctl.pdf")

def plot_hists():
    """
        plot_hists()
        
    Plot precip histograms for past/present. 
    """
    pctls, precip_c, precip_pctls_c, precip_pctls_boot_c, bins_c, hist_c, μ_c, σ_c, k_c, θ_c = load_data(f"data/gpcp_{domain}_{freq}_1979-2000.npz")
    pctls, precip_w, precip_pctls_w, precip_pctls_boot_w, bins_w, hist_w, μ_w, σ_w, k_w, θ_w = load_data(f"data/gpcp_{domain}_{freq}_2000-2020.npz")

    fig, ax = plt.subplots()
    ax.loglog(bins_c[:-1], hist_c, c="tab:blue", label="1979--2000")
    ax.loglog(bins_w[:-1], hist_w, c="tab:orange", label="2000--2020")

    ax.legend()
    ax.set_xlabel("Precipitation (mm day$^{-1}$)")
    ax.set_ylabel("Probability")
    ax.set_xlim([1e-5, 1e3])
    ax.set_ylim([1e-13, 1e15])
    plt.savefig(f"gpcp_{domain}_{freq}_hist.png")
    print(f"gpcp_{domain}_{freq}_hist.png")
    plt.savefig(f"gpcp_{domain}_{freq}_hist.pdf")
    print(f"gpcp_{domain}_{freq}_hist.pdf")
    plt.close()

################################################################################
# run
################################################################################

# percentiles
pctl_min = 0.09
pctl_max = 0.99999
pctls = cmip6_lib.inv_x_transform(np.linspace(cmip6_lib.x_transform(pctl_min), cmip6_lib.x_transform(pctl_max), 100))

# histogram bins
nbins = 1000

# lat slice
latslice = slice(-90, 90)
# latslice = slice(-30, 30)
# latslice = "extrop"

# lon slice
lonslice = slice(0, 360)

# domain
if latslice == slice(-90, 90):
    domain = "global"
elif latslice == slice(-30, 30):
    domain = "trop"
elif latslice == "extrop":
    domain = "extrop"

# path
path = "/export/data1/hgpeterson/gpcp/"

# monthly or daily
# freq = "monthly"
freq = "daily"

# data files
dfiles = path + freq + "/gpcp*.nc"

# get pctls
# get_data(dfiles, pctls, nbins, latslice, lonslice)

# plot pctls
# plot_pctls()

# plot hists
plot_hists()