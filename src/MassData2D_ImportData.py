#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D jet-mass data loader for OT deconvolution experiments.

Reads the ``jet_mass.txt`` and ``jet_mass_SD.txt`` files: each row corresponds
to a prior point x_k, with the first column giving the x_k coordinate and the
remaining columns giving samples z_j(x_k) from the Markov kernel rho_{x_k}.

The data directory defaults to ``<repo>/data`` but can be overridden with the
``OT_UNFOLDING_DATA_DIR`` environment variable.

Created on Tue Feb 25 20:51:30 2025
@author: katycraig
"""

import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M = 2         # number of locations y_i that the true signal could be measured at
Mprime = 3    # number of noise replicates used to construct nu

DoPlots = False

# regularization in binned noise model
epsilon_bin = 1e-40


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _default_data_dir():
    """Resolve the default data directory.

    Priority:
      1. ``OT_UNFOLDING_DATA_DIR`` environment variable, if set.
      2. ``<repo_root>/data`` relative to this file.
      3. Current working directory as a final fallback.
    """
    env = os.environ.get("OT_UNFOLDING_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    candidate = here.parent / "data"
    if candidate.exists():
        return candidate
    return Path.cwd()


def _load_mass_data(data_dir=None):
    """Load the jet-mass data files."""
    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    mass_path = data_dir / "jet_mass.txt"
    sd_path = data_dir / "jet_mass_SD.txt"
    if not mass_path.exists() or not sd_path.exists():
        raise FileNotFoundError(
            f"Could not find jet_mass.txt / jet_mass_SD.txt in {data_dir}. "
            "Set the OT_UNFOLDING_DATA_DIR environment variable or place the "
            "data files there."
        )
    full_data_mass = np.loadtxt(mass_path, delimiter=" ")
    full_data_mass_SD = np.loadtxt(sd_path, delimiter=" ")
    return full_data_mass, full_data_mass_SD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gaussian_density2d(x, y, centerx, centery, variancex, variancey):
    return (
        np.exp(-((x - centerx) ** 2 / (2 * variancex) + (y - centery) ** 2) / (2 * variancey))
        / (2 * np.pi * np.sqrt(variancex * variancey))
    )


def response_matrix_and_locations(data):
    """Build the response matrix and stacked support locations from a data block.

    Each row of ``data`` holds M samples from one Markov kernel; this function
    stacks them into a single 1D array of locations and returns the
    corresponding response matrix.
    """
    num_rows, num_noisy_columns = np.shape(data)
    y_locations = np.zeros(num_rows * num_noisy_columns)
    RT = np.zeros((num_rows, num_rows * num_noisy_columns))
    for row in range(num_rows):
        y_locations[row * num_noisy_columns:(row + 1) * num_noisy_columns] = data[row]
        RT[row, row * num_noisy_columns:(row + 1) * num_noisy_columns] = 1 / num_noisy_columns
    R = np.transpose(RT)
    return y_locations, R


def fixoutliers(A):
    """Clip samples further than 5 std from the mean back to the mean."""
    mask = np.abs(A) > np.abs(np.mean(A)) + 5 * np.abs(np.std(A))
    A[mask] = np.mean(A)
    return A


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def import2DMassDataBinAndUnbin(binNumber, n_prior, n_sigma_true, data_dir=None):
    """Load jet-mass data and assemble the 2D unfolding problem.

    Parameters
    ----------
    binNumber : int or None
        Number of bins along each axis (so total bin count is ``binNumber**2``).
        If ``None``, returns the unbinned problem.
    n_prior : int
        Number of rows of the data to use as prior points.
    n_sigma_true : int
        Number of rows (following the prior rows) to use for the true signal.
    data_dir : path-like or None
        Where to look for the .txt data files.
    """
    full_data_mass, full_data_mass_SD = _load_mass_data(data_dir)

    d = {"dimension": 2}

    # --- response/kernel support and matrix (prior rows) ---
    y_response_mass, R = response_matrix_and_locations(full_data_mass[:n_prior, 1:M + 1])
    y_response_mass_SD, _ = response_matrix_and_locations(full_data_mass_SD[:n_prior, 1:M + 1])
    y_response_mass_SD = fixoutliers(y_response_mass_SD)

    # --- measured y (next n_sigma_true rows) ---
    y_measured_mass, R_truth_mass = response_matrix_and_locations(
        full_data_mass[n_prior:n_prior + n_sigma_true, 1:Mprime + 1]
    )
    y_measured_mass_SD, R_truth_mass_SD = response_matrix_and_locations(
        full_data_mass_SD[n_prior:n_prior + n_sigma_true, 1:Mprime + 1]
    )
    y_measured_mass_SD = fixoutliers(y_measured_mass_SD)

    # --- sigma_0 prior ---
    d["x_prior"] = np.transpose((full_data_mass[:n_prior, 0], full_data_mass_SD[:n_prior, 0]))
    d["sigma_0"] = np.ones(n_prior) / n_prior

    # --- sigma_true (reweighted Gaussian so it differs from sigma_0) ---
    d["x_true"] = np.transpose((
        full_data_mass[n_prior:n_prior + n_sigma_true, 0],
        full_data_mass_SD[n_prior:n_prior + n_sigma_true, 0],
    ))
    d["sigma_true"] = gaussian_density2d(
        d["x_true"][:, 0], d["x_true"][:, 1], 30, -5, 10, 2
    )
    d["sigma_true"] = d["sigma_true"] / np.sum(d["sigma_true"])

    # --- measured distribution nu ---
    d["y_measured_unbinned"] = np.transpose((y_measured_mass, y_measured_mass_SD))
    d["nu_unbinned"] = np.dot(R_truth_mass, d["sigma_true"])

    # --- response matrix and simulated data ---
    d["y_response_unbinned"] = np.transpose((y_response_mass, y_response_mass_SD))
    d["R_unbinned"] = R
    sim_vec = np.dot(d["R_unbinned"], d["sigma_0"])

    if binNumber is None:
        d["y_measured"] = d["y_measured_unbinned"]
        d["y_response"] = d["y_response_unbinned"]
        d["nu"] = d["nu_unbinned"]
        d["R"] = d["R_unbinned"]
        d["sim"] = sim_vec
        return d

    # ---- Binned case ----
    histogram_sim, xedges, yedges = np.histogram2d(
        d["y_response_unbinned"][:, 0], d["y_response_unbinned"][:, 1],
        binNumber, density=True, weights=sim_vec,
    )
    y_response_bin2d = np.meshgrid(
        (xedges[1:] + xedges[:-1]) / 2, np.flip((yedges[1:] + yedges[:-1]) / 2)
    )
    d["y_response"] = np.transpose((
        np.ndarray.flatten(y_response_bin2d[0]),
        np.ndarray.flatten(y_response_bin2d[1]),
    ))
    d["y_measured"] = d["y_response"]

    # Bin indices; x coordinate -> column, y coordinate -> row (flipped)
    y_response_mass_bin_indices = np.digitize(d["y_response_unbinned"][:, 0], xedges)
    y_response_mass_SD_bin_indices = np.digitize(d["y_response_unbinned"][:, 1], np.flip(yedges))
    for idx_arr in (y_response_mass_bin_indices, y_response_mass_SD_bin_indices):
        idx_arr[idx_arr == binNumber + 1] = binNumber
        idx_arr[idx_arr == 0] = 1

    # Bin the response matrix
    Matrix_bin_response = np.zeros((binNumber ** 2, len(d["y_response_unbinned"][:, 0])))
    for i in range(binNumber):
        for j in range(binNumber):
            Matrix_bin_response[i * binNumber + j] = (
                (y_response_mass_bin_indices == (j + 1))
                * (y_response_mass_SD_bin_indices == (i + 1))
            )
    R_bin = np.dot(Matrix_bin_response, d["R_unbinned"])
    d["R"] = (1 - epsilon_bin) * R_bin + epsilon_bin * np.ones_like(R_bin) / R_bin.shape[0]
    d["sim"] = np.dot(d["R"], d["sigma_0"])

    # Bin the measured data
    histogram_nu, _, _ = np.histogram2d(
        y_measured_mass, y_measured_mass_SD, (xedges, yedges),
        density=True, weights=d["nu_unbinned"],
    )
    # flip + transpose because histogram2d indexes (y, x) and top-down
    histogram_nu = np.ndarray.flatten(np.flip(np.transpose(histogram_nu), axis=0))
    histogram_nu = (1 - epsilon_bin) * histogram_nu + epsilon_bin * np.ones_like(histogram_nu) / histogram_nu.size
    d["nu"] = histogram_nu / np.sum(histogram_nu)

    return d
