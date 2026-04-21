# -*- coding: utf-8 -*-
"""
1D toy-data generator for comparing OT deconvolution with IBU / Richardson–Lucy.

Samples three Gaussians for the true distribution sigma_true and applies a
shifted Gaussian noise model to produce measured data. Also builds the
response matrix and optionally bins the data.

Created on Wed Feb  7 21:34:40 2024
@author: bfaktor
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Default 1D problem parameters
# ---------------------------------------------------------------------------
# prior support
n_sigma_0 = 150
prior_LHS, prior_RHS = -1, 1

# noise model: gaussian with variance beta and shift t(x)
beta = 0.1 ** 2


def t(x):
    """Shift function applied before adding Gaussian noise."""
    return x + 0.5 * np.sign(x)


M = 1        # number of simulated measurement replicates per prior point
Mprime = 3   # number of noisy replicates per true point for nu

# true signal: mixture of three Gaussians
c1, c2, c3 = 0.75, -0.75, 0.0
v1, v2, v3 = 0.05, 0.05, 0.05
w1, w2, w3 = 1 / 4, 1 / 2, 1 / 4
n_sigma_true = 150

# regularization used when binning
epsilon_bin = 1e-40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gaussian(x, variance):
    return np.exp(-x ** 2 / (2 * variance)) / np.sqrt(np.pi * (2 * variance))


def gaussian_mixture(y, c1, c2, c3, v1, v2, v3, w1, w2, w3):
    return (
        w1 * gaussian(y - c1, v1)
        + w2 * gaussian(y - c2, v2)
        + w3 * gaussian(y - c3, v3)
    )


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------
def setup1dtoydata(binNumber, samplingSeed):
    """Build the 1D toy unfolding problem for the given bin count and seed.

    Parameters
    ----------
    binNumber : int or None
        If None, the data is kept unbinned. Otherwise, it is histogrammed
        into ``binNumber`` bins.
    samplingSeed : int
        Base random state; four derived states are used internally.

    Returns
    -------
    dict
        Dictionary of arrays defining the unfolding problem.
    """
    rs1 = samplingSeed
    rs2 = rs1 + 1
    rs3 = rs1 + 2
    rs4 = rs1 + 3
    rs5 = rs1 + 4

    d = {"dimension": 1}

    # --- sigma_0, prior ---
    d["x_prior"] = np.linspace(prior_LHS, prior_RHS, n_sigma_0)
    d["sigma_0"] = np.ones_like(d["x_prior"]) / n_sigma_0

    # --- sigma_true ---
    n1 = int(n_sigma_true * w1)
    n2 = int(n_sigma_true * w2)
    n3 = n_sigma_true - n1 - n2
    x_true_1 = norm.rvs(loc=c1, scale=v1, size=n1, random_state=rs1)
    x_true_2 = norm.rvs(loc=c2, scale=v2, size=n2, random_state=rs2)
    x_true_3 = norm.rvs(loc=c3, scale=v3, size=n3, random_state=rs3)
    d["x_true"] = np.concatenate((x_true_1, x_true_2, x_true_3))
    d["sigma_true"] = np.ones(n_sigma_true) / n_sigma_true

    # --- nu (measured) ---
    d["y_measured_unbinned"] = np.array([])
    for _ in range(Mprime):
        noise = norm.rvs(loc=0, scale=np.sqrt(beta),
                         size=np.size(t(d["x_true"])), random_state=rs4)
        d["y_measured_unbinned"] = np.concatenate(
            (d["y_measured_unbinned"], noise + t(d["x_true"]))
        )
        rs4 += 1
    d["nu_unbinned"] = np.ones_like(d["y_measured_unbinned"]) / d["y_measured_unbinned"].size

    # --- response matrix and simulated data ---
    RT = np.zeros((n_sigma_0, n_sigma_0 * M))
    y_response = []
    for x_index in range(n_sigma_0):
        RT[x_index, x_index * M:(x_index + 1) * M] = 1 / M
        y = (norm.rvs(loc=0, scale=np.sqrt(beta), size=M, random_state=rs5)
             + t(d["x_prior"][x_index]) * np.ones(M))
        rs5 += 1
        y_response = np.concatenate((y_response, y))
    d["y_response_unbinned"] = y_response
    d["R_unbinned"] = np.transpose(RT)
    sim_vec = np.dot(d["R_unbinned"], d["sigma_0"])

    if binNumber is None:
        d["y_measured"] = d["y_measured_unbinned"]
        d["y_response"] = d["y_response_unbinned"]
        d["nu"] = d["nu_unbinned"]
        d["R"] = d["R_unbinned"]
        d["sim"] = sim_vec
        return d

    # --- binned path ---
    _, xedges = np.histogram(y_response, binNumber, density=True, weights=sim_vec)
    d["y_measured"] = (xedges[1:] + xedges[:-1]) / 2
    d["y_response"] = d["y_measured"]

    y_response_bin_indices = np.digitize(y_response, xedges)
    # force indices that lie to the right of the histogram back inside
    y_response_bin_indices[y_response_bin_indices == binNumber + 1] = binNumber

    Matrix_bin_response = np.zeros((binNumber, len(y_response)))
    for i in range(binNumber):
        Matrix_bin_response[i] = (y_response_bin_indices == i + 1)
    R_bin = np.dot(Matrix_bin_response, d["R_unbinned"])
    d["R"] = (1 - epsilon_bin) * R_bin + epsilon_bin * np.ones_like(R_bin) / R_bin.shape[0]
    d["sim"] = np.dot(d["R"], d["sigma_0"])

    histogram_nu, _ = np.histogram(
        d["y_measured_unbinned"], xedges, density=True, weights=d["nu_unbinned"]
    )
    histogram_nu = (1 - epsilon_bin) * histogram_nu + epsilon_bin * np.ones_like(histogram_nu) / histogram_nu.size
    d["nu"] = histogram_nu / np.sum(histogram_nu)

    return d
