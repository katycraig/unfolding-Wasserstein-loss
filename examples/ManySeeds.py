#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare OT deconvolution and IBU on 1D toy data across many random seeds.

Run from the repo root with:

    python examples/ManySeeds.py

Writes a pickle (``temp1ddata.pkl``) with the last run and produces a summary
plot of W2^2 vs. iteration with mean +/- std bands across seeds.

Created on Sun Nov 24 09:14:50 2024
@author: katycraig
"""

import sys
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from OTDeconvolutionAlgorithm import runIBUOT
from ToyData1D_ImportData import setup1dtoydata
from PlottingFunctions import plotUnfoldingMethods1d_unbinned  # noqa: F401  (kept for users who uncomment below)


def run_many_seeds(
    samplingSeedArray,
    binVec=np.array([None, 14, 28, 38], dtype=object),
    epsVec=np.array([3e-5]),
    IBU_num_iterations=25,
    OT_num_iterations_original=25,
    W2_stopping=1e-4,
    IBU_stopping=1e-4,
    pickle_path="temp1ddata.pkl",
):
    """Sweep seeds and bin counts, plotting W2^2 vs. iteration."""
    colorVec = ["#028F38", "#0451A9", "#8E090B", "#F18120"] * 3
    linestyleVec = ["-", ":", "--", "-"] * 3
    markerVec = [None, None, None, "x"] * 3
    markeveryVec = [None, None, None, 1] * 3
    hatchstyleVec = [None, ".", "-", "x"] * 3

    plt.rcParams["hatch.linewidth"] = 2.0

    plt.figure("AccuracyAlongIterations", figsize=(5, 4))
    plt.xlabel("Iteration")
    plt.ylabel(r"$W_2^2(\nu_\sigma,\nu)$")
    plt.yscale("log")
    plt.ylim((1e-4, 1e-1))
    plt.xlim((0, OT_num_iterations_original))

    for binIndex, bn in enumerate(binVec):
        if bn is None:
            OT_num_iterations = OT_num_iterations_original
            epsVecIterations = epsVec
        else:
            OT_num_iterations = 0
            epsVecIterations = np.array([0])

        for eps in epsVecIterations:
            W2_IBU_differentSamples, W2_OT_differentSamples = [], []

            for samplingSeed in samplingSeedArray:
                unfoldingInputData = setup1dtoydata(bn, int(samplingSeed))
                unfoldingOutputData = runIBUOT(
                    unfoldingInputData, OT_num_iterations, W2_stopping,
                    IBU_num_iterations, IBU_stopping, eps,
                )
                with open(pickle_path, "wb") as f:
                    pickle.dump(
                        (unfoldingInputData, unfoldingOutputData,
                         OT_num_iterations, IBU_num_iterations, bn),
                        f,
                    )
                
                # Uncomment for visual comparison of OT and RL, as in Figure 4
                # plotUnfoldingMethods1d_unbinned()

                W2_IBU_differentSamples.append(unfoldingOutputData["W2dist_IBUiterations"])
                W2_OT_differentSamples.append(unfoldingOutputData["W2dist_OTiterations"])

            if bn is not None:
                arr = np.array(W2_IBU_differentSamples) ** 2
                mean = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)
                plt.plot(np.arange(len(mean)), mean,
                         linestyle=linestyleVec[binIndex],
                         marker=markerVec[binIndex],
                         markevery=markeveryVec[binIndex],
                         color=colorVec[binIndex],
                         label=r"RL, $n_\mathrm{bin}=$" + str(bn))
                plt.fill_between(np.arange(len(mean)), mean - std, mean + std,
                                 hatch=hatchstyleVec[binIndex],
                                 color=colorVec[binIndex], alpha=0.2)
            else:
                arr = np.array(W2_OT_differentSamples) ** 2
                mean = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)
                plt.plot(np.arange(len(mean)), mean,
                         linestyle=linestyleVec[binIndex],
                         marker=markerVec[binIndex],
                         markevery=markeveryVec[binIndex],
                         color=colorVec[binIndex],
                         label="OT")
                plt.fill_between(np.arange(len(mean)), mean - std, mean + std,
                                 hatch=hatchstyleVec[binIndex],
                                 color=colorVec[binIndex], alpha=0.2)

    plt.legend(loc="upper right", framealpha=1)
    plt.tight_layout()


if __name__ == "__main__":
    # A small seed count so the default invocation stays quick; widen this up
    # (e.g. np.arange(10) or np.arange(100)) for a full study.
    samplingSeedArray = np.linspace(0,100,40,dtype=int)
    run_many_seeds(samplingSeedArray)
    plt.show()
