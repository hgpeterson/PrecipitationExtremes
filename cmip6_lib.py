# imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincinv
from scipy.interpolate import interp1d
import os, re

# plotting stylesheet
plt.style.use("~/paper_plots.mplstyle")

################################################################################
# histogram stuff
################################################################################
"""
    bin_widths, pdf = hist_to_pdf(bins, hist)
    
Convert histogram with number of counts in each bin to pdf (normalize area)
"""
def hist_to_pdf(bins, hist):
    N = np.sum(hist)
    bin_widths = bins[1:] - bins[:-1]
    pdf = 1/N * hist/bin_widths
    return bin_widths, pdf

"""
    pctls, precip_func = hist_to_pctls(bins, hist)
    
Computes percentiles from histogram data and returns interpolation function
"""
def hist_to_pctls(bins, hist):
    # to pdf
    bin_widths, hist = hist_to_pdf(bins, hist)
    
    # percentiles
    pctls = np.cumsum(hist*bin_widths)

    # interpolate
    domain = np.where(np.logical_and(pctls >= 0.01, pctls <= 0.999999))
    x = pctls[domain]
    y = bins[1:][domain]
    precip_func = interp1d(x, y, kind="cubic", bounds_error=False, fill_value=np.nan)
    
    return pctls, precip_func

################################################################################
# gamma distribution stuff
################################################################################

"""
    k, θ = gamma_moments(μ, σ)

Get shape and scale parameters for gamma distribution method of moments fit of data. 
"""
def gamma_moments(μ, σ):
    k = μ**2/σ**2
    θ = σ**2/μ
    return k, θ

"""
    k, θ = gamma_fit_from_hist(bins, hist)
    
Fit a Gamma distribution to binned precip data.
"""
def gamma_fit_from_hist(bins, hist):
    # get mean and std from hists
    μ, σ = get_mean_std_from_hist(bins, hist)
    
    # get k and θ by m.o.m.
    k, θ = gamma_moments(μ, σ)

    return k, θ

"""
    pdf = gamma_dist(x, k, θ)

Return gamma distribution pdf given by parameters k and θ over domain x
"""
def gamma_dist(x, k, θ):
    return x**(k-1)*np.exp(-x/θ)/(gamma(k)*θ**k)

################################################################################
# mean/variance
################################################################################

"""
    μ, σ = get_mean_std(precip)
    
Get the mean and standard deviation of precip data.
"""
def get_mean_std(precip):
    μ = np.mean(precip) # this is why we weight precip by cos(ϕ)
    σ = np.std(precip)
    return μ, σ

"""
    μ, σ = get_mean_std_from_hist(bins, hist)
    
Get the mean and standard deviation of precip data.
"""
def get_mean_std_from_hist(bins, hist):
    bin_widths, hist = hist_to_pdf(bins, hist)
    μ = np.sum(bins[1:]*hist*bin_widths)
    σ = np.sqrt(np.sum(bins[1:]**2*hist*bin_widths))
    return μ, σ

"""
    da_weighted_mean = area_weighted_mean(da)
    
Compute area weighted mean of DataArray object, weighting by cos(ϕ).
"""
def area_weighted_mean(data):
    weights = np.cos(np.deg2rad(data.lat))

    data_weighted = data.weighted(weights)
    data_weighted_mean = data_weighted.mean(("lon", "lat"))

    data_weighted_mean = data_weighted_mean.compute().values
    return data_weighted_mean

################################################################################
# analysis
################################################################################

"""
    file_list = get_file_list(path)
    
Get list of cmip6 files and return it as [source_id, grid_label] array.
"""
def get_file_list(path):
    # make an array of source_id's and grid_label's to choose from
    file_list = np.array(["", ""])
    filenames = os.listdir(path)
    for filename in filenames:
        components = re.split("_", filename)
        source_id = components[2]
        grid_label = components[5]
        file_list = np.vstack((file_list, [source_id, grid_label]))
    file_list = np.unique(file_list[1:, :], axis=0)
    return file_list

"""
    ΔT = get_ΔT(dfiles, latslice, lonslice)
    
Compute the change in mean temperature for ssp585 simulation as difference between first
and last 20 year averages.
"""
def get_ΔT(dfiles, latslice, lonslice):
    ds = xr.open_mfdataset(dfiles)
    tas = ds["tas"] # Near-Surface Air Temp (K)
    if latslice == "extrop":
        T0 = tas.sel(time=slice("2015", "2035"), lon=lonslice).where(np.abs(tas.lat)>=30).mean("time")
        Tf = tas.sel(time=slice("2080", "2100"), lon=lonslice).where(np.abs(tas.lat)>=30).mean("time")
    else:
        T0 = tas.sel(time=slice("2015", "2035"), lat=latslice, lon=lonslice).mean("time")
        Tf = tas.sel(time=slice("2080", "2100"), lat=latslice, lon=lonslice).mean("time")
    ΔT = area_weighted_mean(Tf) - area_weighted_mean(T0)
    return ΔT

"""
    bootstrap_indices = stationary_bootstrap(datasize, samplesize, blocksize, nboot)
    
As in Poltis and Romano 1984: Returns `nboot` x `samplesize` array of indices from 
0 to `datasize` using blocks of size given by geometric distribution of size `blocksize`.
"""
def stationary_bootstrap(datasize, samplesize, blocksize, nboot):
    bootstrap_indices = np.zeros((nboot, samplesize), dtype=int)
    for i in range(nboot):
        # size of sample follows geometric distribution with mean nsamples_mean
        blocksizes = np.random.geometric(1/blocksize, size=datasize) 

        # random starting indices of each block
        indices = np.random.randint(datasize, size=datasize)
        
        # construct pseudodata
        j = 0
        k = 0
        while j < samplesize:
            I = indices[k]
            L = blocksizes[k]
            for n in range(L):
                if j + n == samplesize:
                    break
                bootstrap_indices[i, j + n] = (I + n) % datasize
            j += L
            k += 1
    return bootstrap_indices

################################################################################
# plots
################################################################################

# log-log axes
x_transform = lambda x: -np.log10(1 - x)
inv_x_transform = lambda x: 1 - 10**(-x)
y_transform = lambda y: np.log10(y)

"""
    fix_xticks(ax)
    
Adjust the xticks of axis `ax` to show percentiles following x_transform().
"""
def fix_xticks(ax):
    xticks = np.zeros(10)
    xticklabels = ["9", "90"]
    xticks[0] = 0.09
    xticks[1] = 0.9
    for i in range(2, 10):
        xticks[i] = xticks[i-1] + 9*10**(-i)
        label = "{:0."+ str(i-2) + "f}"
        xticklabels.append(label.format(100*xticks[i]))
    ax.set_xticks(x_transform(xticks))
    ax.set_xlim(x_transform(np.array([0.09, 0.9999])))
    ax.set_xticklabels(xticklabels)
    
"""
    fix_yticks(ax)
    
Adjust the yticks of axis `ax` to show precipitation following y_transform().
"""
def fix_yticks(ax):
    yticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3])
    ax.set_yticks(y_transform(yticks))
    ax.set_yticklabels(["$10^{-1}$", "$10^0$", "$10^1$", "$10^2$", "$10^3$"])
    ax.set_ylim(y_transform(np.array([yticks[0], yticks[-1]])))
    
"""
    axtwin = setup_pctl_plot(ax)
    
Setup axes on axis `ax` for typical percentile plot with twin axis `axtwin` for ratios.
"""
def setup_pctl_plot(ax):
    # ticks
    fix_xticks(ax)
    fix_yticks(ax)

    # ratios on twin axis
    axtwin = ax.twinx()
    axtwin.set_ylim([-10, 30])
    axtwin.tick_params(axis="y", colors="tab:green")
    axtwin.axhline(0, ls="-", c="k", lw=0.5)
    axtwin.spines['right'].set_visible(True)
    axtwin.spines['right'].set_color("tab:green")
    
    return axtwin

"""
    pctl_plot_hist(ax, bins_c, hist_c, k_c, θ_c, bins_w, hist_w, k_w, θ_w)
    
Make a percentile plot from histogram data.
"""
def pctl_plot_hist(ax, ΔT, bins_c, hist_c, k_c, θ_c, bins_w, hist_w, k_w, θ_w):
    # set ticks and twin axis
    axtwin = setup_pctl_plot(ax)
    
    # percentiles for fits
    pctl_min = 0.09
    pctl_max = 0.9999
    pctls = inv_x_transform(np.linspace(x_transform(pctl_min), x_transform(pctl_max), 100))
    
    # gamma fits
    precip_fit_c = θ_c*gammaincinv(k_c, pctls)
    precip_fit_w = θ_w*gammaincinv(k_w, pctls)

    # percentiles for data
    pctls_c, precip_func_c = hist_to_pctls(bins_c, hist_c)
    pctls_w, precip_func_w = hist_to_pctls(bins_w, hist_w)
    
    # plot
    ax.plot(x_transform(pctls_c), y_transform(bins_c[1:]), label="control")
    ax.plot(x_transform(pctls), y_transform(precip_fit_c), c="tab:blue", ls="--")
    ax.plot(x_transform(pctls_w), y_transform(bins_w[1:]), label="warming")
    ax.plot(x_transform(pctls), y_transform(precip_fit_w), c="tab:orange", ls="--")
    
    # compute ratios
#     ratio = 100*(precip_func_w(pctls)/precip_func_c(pctls) - 1)/ΔT
    ratio_fit = 100*(precip_fit_w/precip_fit_c - 1)/ΔT
    
    # ratios on twin axis
#     axtwin.plot(x_transform(pctls), ratio, c="tab:green")
    axtwin.plot(x_transform(pctls), ratio_fit, c="tab:green", ls="--")
    
    return axtwin

"""
    pctl_plot_ratios(axtwin, ΔT, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w, θ_w)
    
Add ratio curves to percentile plot.
"""
def pctl_plot_ratios(axtwin, ΔT, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w, θ_w):
    # bootstrap intervals
    precip_pctls_c_low  = np.percentile(precip_pctls_boot_c, 2.5,  axis=0)
    precip_pctls_c_high = np.percentile(precip_pctls_boot_c, 97.5, axis=0)
    precip_pctls_w_low  = np.percentile(precip_pctls_boot_w, 2.5,  axis=0)
    precip_pctls_w_high = np.percentile(precip_pctls_boot_w, 97.5, axis=0)
    
    # gamma fits
    precip_fit_c = θ_c*gammaincinv(k_c, pctls)
    precip_fit_w = θ_w*gammaincinv(k_w, pctls)

    # compute ratios
    ratio = 100*(precip_pctls_w/precip_pctls_c - 1)/ΔT
    ratio_low = 100*(precip_pctls_w_low/precip_pctls_c_high - 1)/ΔT
    ratio_high = 100*(precip_pctls_w_high/precip_pctls_c_low - 1)/ΔT
    ratio_fit = 100*(precip_fit_w/precip_fit_c - 1)/ΔT

    # ratios on twin axis
    axtwin.plot(x_transform(pctls), ratio, c="tab:green")
    axtwin.fill_between(x_transform(pctls), ratio_low, ratio_high, color="tab:green", alpha=0.5, lw=0)
    axtwin.plot(x_transform(pctls), ratio_fit, c="tab:green", ls="--")

"""
    pctl_plot(ax, ΔT, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w, θ_w)
    
Make a percentile plot!
"""
def pctl_plot(ax, ΔT, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w, θ_w):
    # set ticks and twin axis
    axtwin = setup_pctl_plot(ax)
    
    # bootstrap intervals
    precip_pctls_c_low  = np.percentile(precip_pctls_boot_c, 2.5,  axis=0)
    precip_pctls_c_high = np.percentile(precip_pctls_boot_c, 97.5, axis=0)
    precip_pctls_w_low  = np.percentile(precip_pctls_boot_w, 2.5,  axis=0)
    precip_pctls_w_high = np.percentile(precip_pctls_boot_w, 97.5, axis=0)
    
    # gamma fits
    precip_fit_c = θ_c*gammaincinv(k_c, pctls)
    precip_fit_w = θ_w*gammaincinv(k_w, pctls)

    # pctls and errors
    ax.plot(x_transform(pctls), y_transform(precip_pctls_c), "tab:blue", label="control")
    ax.fill_between(x_transform(pctls), y_transform(precip_pctls_c_low), y_transform(precip_pctls_c_high), alpha=0.5, lw=0)
    ax.plot(x_transform(pctls), y_transform(precip_fit_c), c="tab:blue", ls="--")
    ax.plot(x_transform(pctls), y_transform(precip_pctls_w), "tab:orange", label="warming")
    ax.fill_between(x_transform(pctls), y_transform(precip_pctls_w_low), y_transform(precip_pctls_w_high), alpha=0.5, lw=0)
    ax.plot(x_transform(pctls), y_transform(precip_fit_w), c="tab:orange", ls="--")

    # ratios on twin axis
    pctl_plot_ratios(axtwin, ΔT, pctls, precip_pctls_c, precip_pctls_boot_c, k_c, θ_c, precip_pctls_w, precip_pctls_boot_w, k_w, θ_w)
    
    return axtwin