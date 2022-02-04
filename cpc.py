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
import cartopy.crs as ccrs 
import warnings
import os
warnings.filterwarnings('ignore')
import cmip6_lib

################################################################################
# functions
################################################################################

def get_precip(dfiles, timeslice, latslice, lonslice):
    """
        precip = get_precip(dfiles, timeslice, latslice, lonslice) 

    Load CPC precipitation data.
    """
    # mfdataset for multiple files
    ds = xr.open_mfdataset(dfiles)
    p = ds["precip"] # mm day-1

    # time and space slice
    if latslice == "extrop":
        p = p.sel(time=timeslice, lon=lonslice).where(np.abs(p.lat) >= 30, drop=True)
    else:
        p = p.sel(time=timeslice, lat=latslice, lon=lonslice)

    # turn missing data into nan
    precip = p.values
    precip[np.where(precip < 0)] = np.nan
    # print(f"(number of nans: {np.count_nonzero(np.isnan(precip))})")

    # weight by cos(ϕ)
    cosϕ = np.cos(np.deg2rad(p.lat.values))
    precip *= np.tile(cosϕ, (len(ds.lon), 1)).T

    return precip

def get_pctls(precip, pctls):
    """
        precip_pctls, precip_pctls_boot = get_pctls(precip, pctls)

    Get the precipitation percentiles from monthly GCPC data.
    """
    # compute pctls
    precip_pctls = np.nanpercentile(precip, 100*pctls, interpolation="linear")
    
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
        precip_pctls_boot[i, :] = np.nanpercentile(precip[bootstrap_indices[i, :], :, :], 100*pctls, interpolation="linear")
    
    return precip_pctls, precip_pctls_boot

def get_hist(precip, nbins):
    """
        hist, bins = get_hist(precip, nbins)

    Get the precipitation histogram.
    """
    # setup bins
    pmin = 10**-5
    pmax = 10**3
    bins = np.zeros(nbins)
    bins[1:] = 10**np.linspace(np.log10(pmin), np.log10(pmax), nbins-1) # include 0 as a bin

    # compute pdf
    hist, bins = np.histogram(precip, bins=bins, density=False)    

    return hist, bins

"""
    get_data(dfiles, pctls, nbins, latslice, lonslice)

Save pctls data from gpcp for first and last half of timeseries.
"""
def get_data(dfiles, pctls, nbins, latslice, lonslice):
    # control: first half
    print("computing control pctls.")
    timeslice = slice("1979", "1999")
    print("\tloading precip")
    precip_c = get_precip(dfiles, timeslice, latslice, lonslice)
    print("\tpercentiles")
    precip_pctls_c, precip_pctls_boot_c = get_pctls(precip_c, pctls)
    print("\thist")
    hist_c, bins_c = get_hist(precip_c, nbins)

    # warming: last half
    print("computing warming pctls")
    timeslice = slice("2000", "2020")
    print("\tloading precip")
    precip_w = get_precip(dfiles, timeslice, latslice, lonslice)
    print("\tpercentiles")
    precip_pctls_w, precip_pctls_boot_w = get_pctls(precip_w, pctls)
    print("\thist")
    hist_w, bins_w = get_hist(precip_w, nbins)
    
    # mean and std
    print("computing mean and std")
    μ_c, σ_c = cmip6_lib.get_mean_std(precip_c)
    μ_w, σ_w = cmip6_lib.get_mean_std(precip_w)
    print(f"\t{μ_c}, {σ_c}")
    print(f"\t{μ_w}, {σ_w}")

    np.savez(f"data/cpc_{domain}_first_half.npz", pctls=pctls, precip=precip_c, 
             precip_pctls=precip_pctls_c, precip_pctls_boot=precip_pctls_boot_c, 
             bins=bins_c, hist=hist_c,
             μ=μ_c, σ=σ_c)
    np.savez(f"data/cpc_{domain}_last_half.npz", pctls=pctls, precip=precip_w, 
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

    pctls, precip_c, precip_pctls_c, precip_pctls_boot_c, bins_c, hist_c, μ_c, σ_c, k_c, θ_c = load_data(f"data/cpc_{domain}_first_half.npz")
    pctls, precip_w, precip_pctls_w, precip_pctls_boot_w, bins_w, hist_w, μ_w, σ_w, k_w, θ_w = load_data(f"data/cpc_{domain}_last_half.npz")

    # means and standard deviations from pdfs
    mean_scaling = μ_w/μ_c - 1

    # warming fit based on theory
    extreme_scaling = precip_pctls_w[-1]/precip_pctls_c[-1] - 1
    θ_w_theory = θ_c*(1 + extreme_scaling)
    k_w_theory = k_c*(1 + mean_scaling - extreme_scaling)

    ΔT_global = 1 # no dT for now
    axtwin = cmip6_lib.pctl_plot(ax, ΔT_global, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w_theory, θ_w_theory)

    # label plot
    ax.annotate(f"CPC", (0.76, 0.02), xycoords="axes fraction")
    
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
    label_c = "1979--1999"
    label_w = "2000--2020"
    custom_handles = [label_c, label_w, "ratio", "gamma fit"]
    ax.legend(custom_lines, custom_handles)

    plt.savefig(f"cpc_{domain}_pctl.png")
    print(f"cpc_{domain}_pctl.png")
    plt.close()

def plot_hists():
    """
        plot_hists()
        
    Plot precip histograms for past/present. 
    """
    pctls, precip_c, precip_pctls_c, precip_pctls_boot_c, bins_c, hist_c, μ_c, σ_c, k_c, θ_c = load_data(f"data/cpc_{domain}_first_half.npz")
    pctls, precip_w, precip_pctls_w, precip_pctls_boot_w, bins_w, hist_w, μ_w, σ_w, k_w, θ_w = load_data(f"data/cpc_{domain}_last_half.npz")

    bins_widths_c, hist_c = cmip6_lib.hist_to_pdf(bins_c, hist_c)
    bins_widths_w, hist_w = cmip6_lib.hist_to_pdf(bins_w, hist_w)

    fig, ax = plt.subplots()

    label_c = "1979--1999"
    label_w = "2000--2020"
    ax.loglog(bins_c[:-1], hist_c, c="tab:blue",   label=label_c)
    ax.loglog(bins_w[:-1], hist_w, c="tab:orange", label=label_w)

    ax.legend(loc="lower left")
    ax.set_xlabel("Precipitation (mm day$^{-1}$)")
    # ax.set_ylabel("\# of Events")
    ax.set_ylabel("Probability")
    ax.annotate(f"CPC", (0.8, 0.9), xycoords="axes fraction")
    # ax.set_xlim([1e-2, 1e3])
    # ax.set_ylim([1e-13, 1e15])
    plt.savefig(f"cpc_{domain}_hist.png")
    print(f"cpc_{domain}_hist.png")
    plt.close()

def plot_maps():
    # pctl = 99.9
    pctl = 99.99

    # collect data
    dfile = f"data/cpc_p{pctl}.npz"
    if os.path.isfile(dfile):
        # load it
        d = np.load(dfile)
        lat = d["lat"]
        lon = d["lon"]
        p_c = d["p_c"]
        p_w = d["p_w"]
    else:
        # compute it
        print("loading precip...")
        pctls, precip_c, precip_pctls_c, precip_pctls_boot_c, bins_c, hist_c, μ_c, σ_c, k_c, θ_c = load_data("data/cpc_global_first_half.npz")
        pctls, precip_w, precip_pctls_w, precip_pctls_boot_w, bins_w, hist_w, μ_w, σ_w, k_w, θ_w = load_data("data/cpc_global_last_half.npz")

        print(f"computing {pctl}th percentiles...")
        print("\tcontrol")
        p_c = np.nanpercentile(precip_c, pctl, axis=0)
        print("\twarming")
        p_w = np.nanpercentile(precip_w, pctl, axis=0)

        ds = xr.open_dataset("../../cpc/precip.1979.nc")
        lat = ds.lat.values
        lon = ds.lon.values

        # save it
        print(f"loading {pctl}th percentiles...")
        np.savez(dfile, lat=lat, lon=lon, p_c=p_c, p_w=p_w)
        print(dfile)


    # plot
    fig = plt.figure(figsize=(6.5, 6.5/1.62))
    frac = 0.02
    pad = 0.02

    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.Robinson())
    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.Robinson())
    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.Robinson())

    vmax = 100
    im1 = ax1.pcolormesh(lon, lat, p_c, vmin=0, vmax=vmax, cmap="Blues", transform=ccrs.PlateCarree())
    im2 = ax2.pcolormesh(lon, lat, p_w, vmin=0, vmax=vmax, cmap="Blues", transform=ccrs.PlateCarree())
    plt.colorbar(im1, ax=ax1, label=f"{pctl}th Percentile (mm)", orientation="horizontal", extend="max", fraction=frac, pad=pad)
    plt.colorbar(im2, ax=ax2, label=f"{pctl}th Percentile (mm)", orientation="horizontal", extend="max", fraction=frac, pad=pad)

    r = 100*(p_w/p_c - 1)
    vmax = 30
    im3 = ax3.pcolormesh(lon, lat, r, vmin=-vmax, vmax=vmax, cmap="BrBG", transform=ccrs.PlateCarree())
    plt.colorbar(im3, ax=ax3, label="\% Change", orientation="horizontal", extend="both", fraction=frac, pad=pad)

    ax1.set_title(f"1979--1999")
    ax2.set_title(f"2000--2020")
    ax3.set_title(f"Change")

    ax1.coastlines(lw=0.5)
    ax2.coastlines(lw=0.5)
    ax3.coastlines(lw=0.5)

    fig.savefig(f"cpc_map_p{pctl}.png")
    print(f"cpc_map_p{pctl}.png")
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

# lat slice (lats are reversed for CPC)
# latslice = slice(90, -90)
# latslice = slice(30, -30)
latslice = "extrop"

# lon slice
lonslice = slice(0, 360)

# domain
if latslice == slice(90, -90):
    domain = "global"
elif latslice == slice(30, -30):
    domain = "trop"
elif latslice == "extrop":
    domain = "extrop"

# path
path = "/export/data1/hgpeterson/cpc/"

# data files
dfiles = path + "/precip*.nc"

# # get pctls
# get_data(dfiles, pctls, nbins, latslice, lonslice)

# # plot pctls
# plot_pctls()

# # plot hists
# plot_hists()

# plot maps
plot_maps()