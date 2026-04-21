#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for the OT-deconvolution / IBU comparison.

Notes
-----
- LaTeX rendering is enabled when the ``OT_UNFOLDING_USE_LATEX`` environment
  variable is truthy AND a local LaTeX install is available. Otherwise we
  fall back to mathtext, which works everywhere.
- The matplotlib backend is whatever matplotlib auto-selects unless the user
  sets the ``MPLBACKEND`` environment variable.

Created on Wed May 21 11:14:32 2025
@author: katycraig
"""

import os
import pickle
import shutil
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------
def _latex_available():
    """Return True if both latex and dvipng are on $PATH."""
    return bool(shutil.which("latex")) and bool(shutil.which("dvipng"))


_use_latex = os.environ.get("OT_UNFOLDING_USE_LATEX", "0").lower() in ("1", "true", "yes") \
    and _latex_available()

plt.rcParams.update({
    "text.usetex": _use_latex,
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 20,
    "figure.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})
if _use_latex:
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
mypurple = "#9D3C75"
myyellow = "#47996b"
myblue = "#5076A6"
mypink = "#892961"
myorange = "#c88329"

marker_style = MarkerStyle("|")
marker_style.transform = Affine2D().scale(0.1, 60.0)


# ---------------------------------------------------------------------------
# 1D unbinned plot (reads from pickle)
# ---------------------------------------------------------------------------
def plotUnfoldingMethods1d_unbinned(filename="temp1ddata.pkl"):
    """Plot 1D sigma / nu / prior rows from a pickle produced by ManySeeds."""
    bins_for_visualization = 200
    x_truebw = 0.05
    x_priorbw = 0.05
    y_measured_unbinnedbw = 0.05
    y_response_unbinnedbw = 0.05
    y_responsebw_sim = 0.05

    with open(filename, "rb") as f:
        unfoldingInputData, unfoldingOutputData, OT_num_iterations, IBU_num_iterations, binNumber = \
            pickle.load(f)

    data_LHS = min(
        np.min(unfoldingInputData["x_prior"]),
        np.min(unfoldingInputData["y_measured"]),
        np.min(unfoldingInputData["y_response"]),
    ) - 0.1
    data_RHS = max(
        np.max(unfoldingInputData["x_prior"]),
        np.max(unfoldingInputData["y_measured"]),
        np.max(unfoldingInputData["y_response"]),
    ) + 0.1
    plotx = np.linspace(data_LHS, data_RHS, bins_for_visualization)

    plt.figure(figsize=(5, 7.5))
    kde_gs = GridSpec(
        8, 1, width_ratios=[1],
        height_ratios=[4.5, 1.2, 2.8, 4.5, 1.2, 2.8, 4.5, 1.2], hspace=0,
    )

    # --- sigma ---
    kde_ax1 = plt.subplot(kde_gs[0])
    kde_ax1.set_title(r"Accuracy of $\sigma$")
    sigma_true_kde = gaussian_kde(
        unfoldingInputData["x_true"], bw_method=x_truebw,
        weights=unfoldingInputData["sigma_true"],
    )
    plt.plot(plotx, sigma_true_kde(plotx), color="black", label=r"$\sigma_{\mathrm{true}}$", linewidth=1.5)
    if unfoldingOutputData["IBU_num_iterations"] > 0:
        sigma_IBU_kde = gaussian_kde(
            unfoldingInputData["x_prior"], bw_method=x_truebw,
            weights=unfoldingOutputData["sigma_IBU"],
        )
        plt.plot(plotx, sigma_IBU_kde(plotx), color=myorange, linestyle="dashed", label=r"$\sigma_{RL}$")
    if OT_num_iterations > 0:
        sigma_OT_kde = gaussian_kde(
            unfoldingInputData["x_prior"], bw_method=x_truebw,
            weights=unfoldingOutputData["sigma_OT"],
        )
        plt.plot(plotx, sigma_OT_kde(plotx), color="#6868ac", linestyle="dashed", label=r"$\sigma_{OT}$")
    plt.xlim([-1.6, 1.6])
    plt.xticks([])
    plt.ylim([0, 2.2])
    plt.yticks(np.arange(0, 2.2, 1))
    plt.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0), borderaxespad=0)

    # --- sigma scatter ---
    plt.subplot(kde_gs[1])
    plt.errorbar(
        unfoldingInputData["x_true"], np.ones(np.shape(unfoldingInputData["x_true"])),
        yerr=0.5 * unfoldingInputData["sigma_true"] / np.max(unfoldingInputData["sigma_true"]),
        fmt="none", ecolor="black", elinewidth=1, label=r"$\sigma_{\mathrm{true}}$",
    )
    if unfoldingOutputData["IBU_num_iterations"] > 0:
        plt.errorbar(
            unfoldingInputData["x_prior"], np.zeros(np.shape(unfoldingInputData["x_prior"])),
            yerr=0.5 * np.array(unfoldingOutputData["sigma_IBU"]) / np.max(unfoldingOutputData["sigma_IBU"]),
            fmt="none", ecolor=myorange, elinewidth=1, label=r"$\sigma_{RL}$",
        )
    if OT_num_iterations > 0:
        plt.errorbar(
            unfoldingInputData["x_prior"], np.zeros(np.shape(unfoldingInputData["x_prior"])),
            yerr=0.5 * unfoldingOutputData["sigma_OT"] / np.max(unfoldingOutputData["sigma_OT"]),
            fmt="none", ecolor="#6868ac", elinewidth=1, label=r"$\sigma_{OT}$",
        )
    plt.xlim(kde_ax1.get_xlim())
    plt.ylim([-0.4, 1.4])
    plt.yticks([])

    # --- nu ---
    ax3 = plt.subplot(kde_gs[3])
    ax3.set_title(r"Accuracy of $\nu$")
    nu_kde_unbinned = gaussian_kde(
        unfoldingInputData["y_measured_unbinned"], bw_method=y_measured_unbinnedbw,
        weights=unfoldingInputData["nu_unbinned"],
    )
    plt.plot(plotx, nu_kde_unbinned(plotx), color="black", label=r"$\nu$", linewidth=1.5)
    if unfoldingOutputData["IBU_num_iterations"] > 0:
        nu_IBU_kde = gaussian_kde(
            unfoldingInputData["y_response_unbinned"], bw_method=y_response_unbinnedbw,
            weights=unfoldingOutputData["noisy_sigma_IBU_unbinned"],
        )
        plt.plot(plotx, nu_IBU_kde(plotx), color=myorange, linestyle="dashed",
                 label=r"$\nu_{\sigma_\mathrm{RL}}$")
    if OT_num_iterations > 0:
        nu_OT_kde = gaussian_kde(
            unfoldingInputData["y_response_unbinned"], bw_method=y_response_unbinnedbw,
            weights=unfoldingOutputData["noisy_sigma_OT_unbinned"],
        )
        plt.plot(plotx, nu_OT_kde(plotx), color="#6868ac", linestyle="dashed",
                 label=r"$\nu_{\sigma_\mathrm{OT}}$")
    plt.xlim(kde_ax1.get_xlim())
    plt.xticks([])
    plt.ylim(kde_ax1.get_ylim())
    plt.yticks(kde_ax1.get_yticks())
    plt.legend(loc="best", frameon=False, borderaxespad=0)

    # --- nu scatter ---
    plt.subplot(kde_gs[4])
    plt.errorbar(
        unfoldingInputData["y_measured_unbinned"],
        np.ones(np.shape(unfoldingInputData["y_measured_unbinned"])),
        yerr=0.5 * unfoldingInputData["nu_unbinned"] / np.max(unfoldingInputData["nu_unbinned"]),
        fmt="none", ecolor="black", elinewidth=1,
    )
    if unfoldingOutputData["IBU_num_iterations"] > 0:
        plt.errorbar(
            unfoldingInputData["y_response_unbinned"],
            np.zeros(np.shape(unfoldingInputData["y_response_unbinned"])),
            yerr=0.5 * unfoldingOutputData["noisy_sigma_IBU_unbinned"]
            / np.max(unfoldingOutputData["noisy_sigma_IBU_unbinned"]),
            fmt="none", ecolor=myorange, elinewidth=1,
        )
    if OT_num_iterations > 0:
        plt.errorbar(
            unfoldingInputData["y_response_unbinned"],
            np.zeros(np.shape(unfoldingInputData["y_response_unbinned"])),
            yerr=0.5 * unfoldingOutputData["noisy_sigma_OT_unbinned"]
            / np.max(unfoldingOutputData["noisy_sigma_OT_unbinned"]),
            fmt="none", ecolor="#6868ac", elinewidth=1,
        )
    plt.xlim(kde_ax1.get_xlim())
    plt.ylim([-0.4, 1.4])
    plt.yticks([])

    # --- prior ---
    ax5 = plt.subplot(kde_gs[6])
    ax5.set_title(r"Initialization of $\sigma$ and $\nu$")
    sigma0_kde = gaussian_kde(
        unfoldingInputData["x_prior"], bw_method=x_priorbw,
        weights=unfoldingInputData["sigma_0"],
    )
    plt.plot(plotx, sigma0_kde(plotx), color=mypink, label=r"$\sigma_0$")
    sim_kde = gaussian_kde(
        unfoldingInputData["y_response"], bw_method=y_responsebw_sim,
        weights=unfoldingInputData["sim"],
    )
    plt.plot(plotx, sim_kde(plotx), color=myblue, label=r"$\nu_{\sigma_0}$")
    plt.xlim(kde_ax1.get_xlim())
    plt.xticks([])
    plt.ylim(kde_ax1.get_ylim())
    plt.yticks(kde_ax1.get_yticks())
    plt.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0), borderaxespad=0)

    # --- prior scatter ---
    plt.subplot(kde_gs[7])
    plt.errorbar(
        unfoldingInputData["x_prior"], 0.2 * np.ones(np.shape(unfoldingInputData["x_prior"])),
        yerr=0.25 * unfoldingInputData["sigma_0"] / np.max(unfoldingInputData["sigma_0"]),
        fmt="none", ecolor=mypink, elinewidth=1,
    )
    plt.errorbar(
        unfoldingInputData["y_response"], 0.8 * np.ones(np.shape(unfoldingInputData["y_response"])),
        yerr=0.25 * unfoldingInputData["sim"] / np.max(unfoldingInputData["sim"]),
        fmt="none", ecolor=myblue, elinewidth=1,
    )
    plt.xlim(kde_ax1.get_xlim())
    plt.ylim([-0.2, 1.2])
    plt.yticks([])

    plt.subplots_adjust(hspace=0)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# 1D binned plot
# ---------------------------------------------------------------------------
def plotUnfoldingMethods1d(unfoldingInputData, unfoldingOutputData,
                           OT_num_iterations, IBU_num_iterations, bins_for_visualization):
    data_LHS = min(
        np.min(unfoldingInputData["x_prior"]),
        np.min(unfoldingInputData["y_measured"]),
        np.min(unfoldingInputData["y_response"]),
    ) - 0.1
    data_RHS = max(
        np.max(unfoldingInputData["x_prior"]),
        np.max(unfoldingInputData["y_measured"]),
        np.max(unfoldingInputData["y_response"]),
    ) + 0.1
    bins = np.linspace(data_LHS, data_RHS, bins_for_visualization)

    plt.figure(figsize=(5, 5))
    gs = GridSpec(4, 1, width_ratios=[1], height_ratios=[3.5, 3.5, 3.5, 1])

    ax1 = plt.subplot(gs[0])
    sigma_true_hist, _, _ = plt.hist(
        unfoldingInputData["x_true"], bins, density=True,
        weights=unfoldingInputData["sigma_true"], histtype="step",
        color="black", label=r"$\sigma_{\mathrm{true}}$", linewidth=1.5,
    )
    sigma_RL_hist = sigma_OT_hist = None
    if IBU_num_iterations > 0:
        sigma_RL_hist, _, _ = plt.hist(
            unfoldingInputData["x_prior"], bins, density=True,
            weights=unfoldingOutputData["sigma_IBU"], histtype="step",
            color="orange", linestyle="dashed", label=r"$\sigma_{RL}$",
        )
    if OT_num_iterations > 0:
        sigma_OT_hist, _, _ = plt.hist(
            unfoldingInputData["x_prior"], bins, density=True,
            weights=unfoldingOutputData["sigma_OT"], histtype="step",
            color="#8e64d1", linestyle="dashed",
            label=r"$\sigma_{OT}(" + str(OT_num_iterations) + r"), T=" + str(unfoldingOutputData["T_OT"]) + "$",
        )
    plt.legend(frameon=False)

    plt.subplot(gs[1])
    plt.hist(unfoldingInputData["y_measured"], bins, density=True,
             weights=unfoldingInputData["nu"], histtype="step",
             color="black", label=r"$\nu$", linewidth=1.5)
    if OT_num_iterations > 0:
        plt.hist(unfoldingInputData["y_response"], bins, density=True,
                 weights=unfoldingOutputData["noisy_sigma_OT"], histtype="step",
                 color="#8e64d1", linestyle="dashed", label=r"$\nu_{\sigma_\mathrm{OT}}$")
    if IBU_num_iterations > 0:
        plt.hist(unfoldingInputData["y_response"], bins, density=True,
                 weights=unfoldingOutputData["noisy_sigma_IBU"], histtype="step",
                 color="orange", linestyle="dashed", label=r"$\nu_{\sigma_\mathrm{RL}}$")
    plt.ylim(ax1.get_ylim())
    plt.legend(frameon=False)

    plt.subplot(gs[2])
    plt.hist(unfoldingInputData["x_prior"], bins, density=True,
             weights=unfoldingInputData["sigma_0"], histtype="step",
             color="black", label=r"$\sigma$")
    plt.hist(unfoldingInputData["y_response"], bins, density=True,
             weights=unfoldingInputData["sim"], histtype="step",
             color="blue", label=r"$\nu_{\sigma_0}$")
    plt.ylim(ax1.get_ylim())
    plt.legend(frameon=False)

    plt.subplot(gs[3])
    if IBU_num_iterations > 0 and sigma_RL_hist is not None:
        RL_ratio = np.ones_like(sigma_true_hist)
        mask = sigma_true_hist > 1e-10
        RL_ratio[mask] = sigma_RL_hist[mask] / sigma_true_hist[mask]
        plt.hist(bins[:-1], bins, weights=RL_ratio, histtype="step",
                 color="orange", linestyle="solid")
    if OT_num_iterations > 0 and sigma_OT_hist is not None:
        OT_ratio = np.ones_like(sigma_true_hist)
        mask = sigma_true_hist > 1e-10
        OT_ratio[mask] = sigma_OT_hist[mask] / sigma_true_hist[mask]
        plt.hist(bins[:-1], bins, weights=OT_ratio, histtype="step",
                 color="#8e64d1", linestyle="solid")
    plt.hist(bins[:-1], bins, weights=np.ones_like(sigma_true_hist),
             histtype="step", color="black", linestyle=":")
    plt.ylabel("Extracted/Truth")
    plt.xlabel("Value")
    plt.ylim([0, 2])
    plt.tight_layout()


# ---------------------------------------------------------------------------
# 2D unfolding plots
# ---------------------------------------------------------------------------
def plotUnfoldingMethods2d(binIndex, filename="tempdata.pkl"):
    with open(filename, "rb") as f:
        binNumberVec, inputDataList, outputDataList = pickle.load(f)

    sigma_bins = 30
    nu_bins = 60 if binNumberVec[binIndex] is None else binNumberVec[binIndex]

    inputData, outputData = inputDataList[binIndex], outputDataList[binIndex]

    x_prior = inputData["x_prior"]
    x_true = inputData["x_true"]
    y_measured = inputData["y_measured"]
    y_response = inputData["y_response"]
    nu = inputData["nu"]
    sim = inputData["sim"]
    noisy_sigma_OT = outputData["noisy_sigma_OT"]
    noisy_sigma_IBU = outputData["noisy_sigma_IBU"]
    sigma_0 = inputData["sigma_0"]
    sigma_true = inputData["sigma_true"]
    that_IBU = outputData["sigma_IBU"]
    that_OT = outputData["sigma_OT"]
    OT_num_iterations = outputData["OT_num_iterations"]
    IBU_num_iterations = outputData["IBU_num_iterations"]
    T_OT = outputData["T_OT"]
    T_RL = outputData["T_LR"]

    if that_IBU is None:
        plt.figure(figsize=(9, 5))
        gs = GridSpec(2, 3, width_ratios=[7, 7, 7], height_ratios=[3.5, 3.5])
    else:
        plt.figure(figsize=(12, 5))
        gs = GridSpec(2, 4, width_ratios=[7, 7, 7, 7], height_ratios=[3.5, 3.5])

    plt.subplot(gs[0, 1])
    _, xedges, yedges, _ = plt.hist2d(
        x_true[:, 0], x_true[:, 1], sigma_bins, density=True, weights=sigma_true
    )
    plt.title(r"$\sigma_{\mathrm{true}}$")

    plt.subplot(gs[0, 2])
    plt.hist2d(x_prior[:, 0], x_prior[:, 1], [xedges, yedges],
               density=True, weights=that_OT)
    plt.title(r"$\sigma_{OT}(" + str(OT_num_iterations) + r"), T=" + str(T_OT) + "$")

    if that_IBU is not None:
        plt.subplot(gs[0, 3])
        plt.hist2d(x_prior[:, 0], x_prior[:, 1], [xedges, yedges],
                   density=True, weights=that_IBU)
        plt.title(r"$\sigma_{IBU}(" + str(IBU_num_iterations) + r"), T=" + str(T_RL) + "$")

    plt.subplot(gs[0, 0])
    plt.hist2d(x_prior[:, 0], x_prior[:, 1], [xedges, yedges],
               density=True, weights=sigma_0)
    plt.title(r"$\sigma_\mathrm{prior}$")

    plt.subplot(gs[1, 1])
    h, xedges, yedges, _ = plt.hist2d(
        y_measured[:, 0], y_measured[:, 1], nu_bins, density=True, weights=nu
    )
    plt.title(r"$\nu$")

    # Zoom in on the region where histogram values are above a threshold
    threshold = np.max(h) * 0.01
    y_mask = np.any(h > threshold, axis=0)
    y_indices = np.where(y_mask)[0]
    ylim_low = yedges[y_indices[0]]
    ylim_high = yedges[y_indices[-1] + 1]
    plt.ylim(ylim_low, ylim_high)

    plt.subplot(gs[1, 2])
    plt.hist2d(y_response[:, 0], y_response[:, 1], [xedges, yedges],
               density=True, weights=noisy_sigma_OT)
    plt.title(r"$\nu_{\sigma_\mathrm{OT}}$")
    plt.ylim(ylim_low, ylim_high)

    if that_IBU is not None:
        plt.subplot(gs[1, 3])
        plt.hist2d(y_response[:, 0], y_response[:, 1], [xedges, yedges],
                   density=True, weights=noisy_sigma_IBU)
        plt.title(r"$\nu_{\sigma_\mathrm{IBU}}$")
        plt.ylim(ylim_low, ylim_high)

    plt.subplot(gs[1, 0])
    plt.hist2d(y_response[:, 0], y_response[:, 1], [xedges, yedges],
               density=True, weights=sim)
    plt.title(r"$\nu_{\sigma_0}$")
    plt.ylim(ylim_low, ylim_high)

    plt.tight_layout()


def meanvar2d(positions, weights):
    """Return [<x1>, <x2>, var, <x1/x2>] as a summary of a weighted 2D point cloud."""
    if weights is None:
        return None
    mean = np.sum(positions * np.asarray(weights)[:, None], 0)
    variance = np.sum(
        ((positions[:, 0] - mean[0]) ** 2 + (positions[:, 1] - mean[1]) ** 2) * weights
    )
    ratio = np.sum((positions[:, 0] / positions[:, 1]) * weights)
    return np.array([mean[0], mean[1], variance, ratio])




def plot2DMassSummaryObservablesDifferentBins(filename="tempdata.pkl",Mvalue=2):
    with open(filename, "rb") as f:
        binNumberVec, inputDataList, outputDataList = pickle.load(f)

    with open(filename, 'rb') as f: # Load from file
        binNumberVec, inputDataList, outputDataList = pickle.load(f)

    sigma_IBU_list = []
    justBinNumbers = np.array([])
    for index in range(len(binNumberVec)):
        if binNumberVec[index] is not None:
            justBinNumbers = np.append(justBinNumbers,binNumberVec[index]**2)
            sigma_IBU_list.append(meanvar2d(inputDataList[index]["x_prior"],outputDataList[index]["sigma_IBU"]))
        elif binNumberVec[index] is None:
             sigma_true_unbinned = meanvar2d(inputDataList[index]["x_true"],inputDataList[index]["sigma_true"])
             sigma_OT_unbinned = meanvar2d(inputDataList[index]["x_prior"],outputDataList[index]["sigma_OT"])

    sigma_IBU_binned = np.vstack(sigma_IBU_list)      # shape: (num_bins, 4)

    #plot observables for sigma
    fig, axs = plt.subplots(4, 1,sharex=True, figsize=(3.5,8))
    axs[0].set_title(r'$M='+str(Mvalue)+'$')
    constantLine = np.linspace(justBinNumbers[0],justBinNumbers[-1],20)

    for observableIndex in range(4):
        #label axes according to observable
        if observableIndex == 0:
            axs[0].set_ylabel(r'$\langle x_1 \rangle$')
        if observableIndex == 1:
            axs[1].set_ylabel(r'$\langle x_2 \rangle$')
        if observableIndex == 2:
            axs[2].set_ylabel(r'$\rm{Var}$')
        if observableIndex == 3:
            axs[3].set_ylabel(r'$\langle x_1/x_2 \rangle$')

        axs[observableIndex].plot(constantLine,np.ones(20)*sigma_true_unbinned[observableIndex],color=mypink, linewidth=3, label = r'$\sigma_{\rm true}$') #plot true
        axs[observableIndex].plot(constantLine,np.ones(20)*sigma_OT_unbinned[observableIndex],color=myyellow, label = r'$\sigma_{\rm OT}$') #plot OT
        axs[observableIndex].plot(justBinNumbers,sigma_IBU_binned[:,observableIndex],'.',label = r'$\sigma_{\rm RL}$',color=myorange,linestyle='dashed') #plot binned IBU

    axs[3].set_xlabel('number of bins')
    axs[3].set_xticks([300,600,900])
    ordered_labels = [r'$\sigma_{\rm true}$', r'$\sigma_{\rm OT}$', r'$\sigma_{\rm RL}$']
    handles, labels = axs[3].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ordered_handles = [by_label[label] for label in ordered_labels]
    plt.legend(ordered_handles, ordered_labels,loc='lower center', bbox_to_anchor=(0.34, -1.8), ncol=1)

    fig.align_ylabels()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)


def plotW2distanceAlongIterations(filename):
    with open(filename, "rb") as f:
        binNumberVec, inputDataList, outputDataList = pickle.load(f)

    plt.figure("AccuracyAlongIterations", figsize=(4, 4))
    plt.xlabel("Iteration")
    plt.ylabel(r"$W_2^2(\nu_\sigma,\nu)$")
    plt.yscale("log")
    colorVec = ["#028F38", "#0451A9", "#8E090B", "#F18120"] * 3

    for binIndex in range(len(binNumberVec)):
        outputData = outputDataList[binIndex]
        OT_num_iterations = outputData["OT_num_iterations"]
        IBU_num_iterations = outputData["IBU_num_iterations"]
        if OT_num_iterations and OT_num_iterations > 0:
            plt.plot(range(OT_num_iterations),
                     outputData["W2dist_OTiterations"] ** 2,
                     color=colorVec[binIndex], label="OT")
        if IBU_num_iterations and IBU_num_iterations > 0:
            plt.plot(range(IBU_num_iterations),
                     outputData["W2dist_IBUiterations"] ** 2,
                     linestyle="--", color=colorVec[binIndex],
                     label=r"RL, $n_\mathrm{bin}=$" + str(binNumberVec[binIndex] ** 2))

    plt.legend(loc="upper right")
    plt.tight_layout()
