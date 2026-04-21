#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare OT deconvolution and IBU on the 2D jet-mass data for different binnings.

Run from the repo root with:

    python examples/MassData2D_Examples.py

Outputs a pickle (``tempdata.pkl``) and summary plots.

Created on Sun Nov 24 09:14:50 2024
@author: katycraig
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# make src/ importable whether run from repo root or examples/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from OTDeconvolutionAlgorithm import runIBUOT
from MassData2D_ImportData import import2DMassDataBinAndUnbin
from PlottingFunctions import (
    plot2DMassSummaryObservablesDifferentBins,
    plotUnfoldingMethods2d,
    plotW2distanceAlongIterations,
)


def unfolding2DMassDataForDifferentBinnings(
    binNumberVec, IBU_num_iterations, OT_num_iterations_original,
    n_prior, n_true, W2_stopping, IBU_stopping, eps,
    output_pickle="tempdata.pkl",
):
    """Run IBU/OT on the 2D mass data over a grid of bin counts.

    For ``binNumber is None`` (unbinned), OT deconvolution is run with
    ``OT_num_iterations_original`` iterations; IBU is skipped because it
    requires a binned grid. For binned runs, only IBU is executed.
    """
    startTime = time.time()
    inputDataList, outputDataList = [], []

    for bn in binNumberVec:
        if bn is None:
            OT_num_iterations = OT_num_iterations_original
        else:
            OT_num_iterations = 0
        unfoldingInputData = import2DMassDataBinAndUnbin(bn, n_prior, n_true)
        unfoldingOutputData = runIBUOT(
            unfoldingInputData, OT_num_iterations, W2_stopping,
            IBU_num_iterations, IBU_stopping, eps,
        )
        inputDataList.append(unfoldingInputData)
        outputDataList.append(unfoldingOutputData)

    with open(output_pickle, "wb") as f:
        pickle.dump((binNumberVec, inputDataList, outputDataList), f)

    print(f"Total time = {time.time() - startTime:.1f}s")


if __name__ == "__main__":
    # Full bin sweep used in the paper. This is expensive (each bin count
    # builds a (binNumber**2 x n_prior) response matrix and runs 30 IBU
    # iterations); shorten to e.g. np.array([None, 8, 12, 16], dtype=object)
    # for a quicker smoke test.
    binNumberVec = np.array(
        [None,8,10,14,18,22,26,30], dtype=object
    )
    IBU_num_iterations = 30
    OT_num_iterations_original = 30
    W2_stopping = 1e-4
    IBU_stopping = W2_stopping
    n_prior = 100
    n_true = n_prior
    eps = 3e-5

    unfolding2DMassDataForDifferentBinnings(
        binNumberVec, IBU_num_iterations, OT_num_iterations_original,
        n_prior, n_true, W2_stopping, IBU_stopping, eps,
    )

    plot2DMassSummaryObservablesDifferentBins()
    
    # Uncomment to also produce a 2D unfolding visualization for a specific run:
    # plotUnfoldingMethods2d(0)
    # plotW2distanceAlongIterations('tempdata.pkl')

    plt.show()
