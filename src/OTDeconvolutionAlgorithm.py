# -*- coding: utf-8 -*-
"""
Optimal Transport (OT) Deconvolution and Iterative Bayesian Unfolding (IBU / RL).

This module implements the generalized Sinkhorn / Douglas–Rachford algorithm
for OT-based deconvolution, as well as a reference implementation of
Iterative Bayesian Unfolding (equivalently Richardson–Lucy) for comparison.

Created on Wed Feb  7 21:34:40 2024
@author: bfaktor
"""

import time
import numpy as np
import ot
from scipy.special import wrightomega, logsumexp


# ---------------------------------------------------------------------------
# Algorithm hyperparameters
# ---------------------------------------------------------------------------
prior_eps_reg = 0.01            # entropic regularization for initial Sinkhorn warm-start
OT_prior_num_iterations = 100   # number of Sinkhorn iterations for the warm start
num_DR_iterations = 25          # Douglas–Rachford iterations used inside Prox_F
tau = 0.001                     # Douglas–Rachford step size


# ---------------------------------------------------------------------------
# Core proximal operator (Douglas–Rachford) used inside generalized Sinkhorn
# ---------------------------------------------------------------------------
def Prox_F_Bauschke(project_kernel_B_matrix, logvec):
    """Proximal operator of F via Douglas-Rachford splitting (Bauschke et al.)."""
    x_new = np.exp(logvec)
    z_new = np.exp(logvec)

    for _ in range(num_DR_iterations):
        x_old = x_new
        y_new = project_kernel_B_matrix @ x_old
        wrightomega_arg = logvec - np.log(tau) + (2 * y_new - x_old) / tau
        z_new = tau * np.real(wrightomega(wrightomega_arg))
        x_new = x_old + z_new - y_new

    # prevent zeros in z_new (they would cause -inf in log)
    nonzero = z_new[z_new != 0]
    if nonzero.size > 0:
        z_new[z_new == 0] = np.min(nonzero)
    return z_new


# ---------------------------------------------------------------------------
# Warm-start Sinkhorn
# ---------------------------------------------------------------------------
def sinkhorn_iterations(firstmarg, secondmarg, C_normalized,
                        prior_eps_reg, OT_prior_num_iterations, C_norm_constant):
    """Standard log-domain Sinkhorn iterations to solve OT(firstmarg, secondmarg)."""
    CT = np.transpose(C_normalized)
    logv = np.zeros(len(secondmarg))  # arbitrary starting guess

    for _ in range(OT_prior_num_iterations):
        logKv = logsumexp(-C_normalized / prior_eps_reg + logv, 1)
        logu = np.log(firstmarg) - logKv
        logKTu = logsumexp(-CT / prior_eps_reg + logu, 1)
        logv = np.log(secondmarg) - logKTu

    return logu, logv


# ---------------------------------------------------------------------------
# Generalized Sinkhorn / OT deconvolution iterations
# ---------------------------------------------------------------------------
def generalized_sinkhorn_iterations_with_input_sigma_0(
        OT_num_iterations, b_vec, K_tilde, project_kernel_B_matrix,
        unfoldingInputData, logu, eps,
        cost_unbinned_normalized, cost_unbinned, W2_stopping):
    """Generalized Sinkhorn-like iterations for the deconvolution problem."""
    CT = np.transpose(cost_unbinned_normalized)
    p = len(unfoldingInputData["x_prior"])
    n = len(b_vec) - 1
    W2dist_OTiterations = []
    iternum = 0

    while iternum < OT_num_iterations:
        iternum += 1

        logfirstu, logsecondu = logu[:-p], logu[-p:]
        firstlogK_tildeTu = logsumexp(-CT / eps + logfirstu, 1)
        secondlogK_tildeTu = logsumexp(logsecondu)
        logK_tildeTu = np.hstack((firstlogK_tildeTu, secondlogK_tildeTu))

        logv = np.log(b_vec) - logK_tildeTu
        logfirstv, logsecondv = logv[:n], logv[-1:]

        firstlogK_tildev = logsumexp(-cost_unbinned_normalized / eps + logfirstv, 1)
        secondlogK_tildev = logsecondv * np.ones(p)
        logK_tildev = np.hstack((firstlogK_tildev, secondlogK_tildev))
        Prox = Prox_F_Bauschke(project_kernel_B_matrix, logu + logK_tildev)

        logu = np.log(Prox) - logK_tildev

        # early-stop based on W2 distance to measured distribution
        currentLogSigmaApprox = logu[-p:] + logv[-1] * np.ones(np.shape(logu[-p:]))
        normalizedsigma = np.exp(currentLogSigmaApprox) / np.sum(np.exp(currentLogSigmaApprox))
        noisy_sigma_OT = np.dot(unfoldingInputData["R_unbinned"], normalizedsigma)
        W2dist_numeasuredOT = ot.emd2(
            noisy_sigma_OT, unfoldingInputData["nu_unbinned"],
            cost_unbinned, numItermax=int(1e6)
        ) ** 0.5
        W2dist_OTiterations = np.append(W2dist_OTiterations, W2dist_numeasuredOT)

        if W2dist_numeasuredOT < W2_stopping:
            print(f"W2 distance small enough; stopping early at iteration {iternum}")
            break

    logP_updated = logu[-p:] + logv[-1] * np.ones(np.shape(logu[-p:]))
    return logP_updated, W2dist_OTiterations, iternum


# ---------------------------------------------------------------------------
# Iterative Bayesian Unfolding (equivalently Richardson–Lucy)
# ---------------------------------------------------------------------------
def IBU(unfoldingInputData, IBU_num_iterations, IBU_stopping):
    """Iterative Bayesian Unfolding / Richardson–Lucy."""
    cost_unbinned = costMatrices_binned_and_unbinned(unfoldingInputData)

    W2dist_IBUiterations = []
    current_sigma = unfoldingInputData["sigma_0"]
    q = 0
    while q < IBU_num_iterations:
        q += 1
        noisy_sigma_IBU = np.dot(unfoldingInputData["R_unbinned"], current_sigma)
        W2dist_numeasuredIBU = ot.emd2(
            noisy_sigma_IBU, unfoldingInputData["nu_unbinned"],
            cost_unbinned, numItermax=int(1e6)
        ) ** 0.5
        W2dist_IBUiterations = np.append(W2dist_IBUiterations, W2dist_numeasuredIBU)
        if W2dist_numeasuredIBU < IBU_stopping:
            print(f"IBU distance small enough; stopping early at iteration {q}")
            break

        sim = unfoldingInputData["R"] @ current_sigma
        current_sigma = np.sum(
            unfoldingInputData["R"] * current_sigma[None, :]
            * unfoldingInputData["nu"][:, None] / sim[:, None],
            axis=0,
        )

    return current_sigma, W2dist_IBUiterations, q


# ---------------------------------------------------------------------------
# Cost matrix construction
# ---------------------------------------------------------------------------
def costMatrices_binned_and_unbinned(unfoldingInputData):
    """Build the squared-Euclidean cost matrix between response and measured locations."""
    if unfoldingInputData["dimension"] == 1:
        # shape (m, n) = (len(y_response), len(y_measured)), to match the
        # 2D convention that the rest of OT_Deconvolve assumes.
        cost_unbinned = (
            unfoldingInputData["y_response_unbinned"][:, None]
            - unfoldingInputData["y_measured_unbinned"][None, :]
        ) ** 2
    elif unfoldingInputData["dimension"] == 2:
        yr1 = unfoldingInputData["y_response_unbinned"][:, 0]
        yr2 = unfoldingInputData["y_response_unbinned"][:, 1]
        ym1 = unfoldingInputData["y_measured_unbinned"][:, 0]
        ym2 = unfoldingInputData["y_measured_unbinned"][:, 1]
        cost_unbinned = (yr1[:, None] - ym1[None, :]) ** 2 + (yr2[:, None] - ym2[None, :]) ** 2
    else:
        raise ValueError(f"Unsupported dimension: {unfoldingInputData['dimension']}")
    return cost_unbinned


# ---------------------------------------------------------------------------
# End-to-end OT deconvolution
# ---------------------------------------------------------------------------
def OT_Deconvolve(OT_num_iterations, W2_stopping, unfoldingInputData, eps):
    """Run OT deconvolution on the given input data.

    Note
    ----
    This routine is designed to run on *unbinned* data: i.e. the setting where
    ``y_measured == y_measured_unbinned`` and ``y_response == y_response_unbinned``.
    The accompanying example scripts skip OT on binned inputs (setting
    ``OT_num_iterations = 0`` for binned runs), so this has historically been
    the only tested path.
    """
    # Guard against the half-implemented binned-OT path (y_response has been
    # replaced by bin centers while cost_unbinned still uses the original
    # unbinned locations, which gives a shape mismatch in K_tilde).
    is_unbinned = (
        len(unfoldingInputData["y_response"]) == len(unfoldingInputData["y_response_unbinned"])
        and len(unfoldingInputData["y_measured"]) == len(unfoldingInputData["y_measured_unbinned"])
    )
    if not is_unbinned:
        raise NotImplementedError(
            "OT_Deconvolve currently supports only unbinned inputs. "
            "For binned data, use IBU (set OT_num_iterations=0)."
        )

    n = len(unfoldingInputData["y_measured"])
    p = len(unfoldingInputData["x_prior"])
    m = len(unfoldingInputData["y_response"])

    cost_unbinned = costMatrices_binned_and_unbinned(unfoldingInputData)
    C_norm_constant = np.max(cost_unbinned)
    cost_unbinned_normalized = cost_unbinned / C_norm_constant

    B = np.hstack((np.identity(m), -unfoldingInputData["R"]))
    b_vec = np.append(unfoldingInputData["nu"], [1])
    K_tilde = np.block([
        [np.exp(-cost_unbinned_normalized / eps), np.zeros((m, 1))],
        [np.zeros((p, n)), np.ones((p, 1))],
    ])

    Q, _ = np.linalg.qr(B.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        project_onto_range = Q @ Q.T  # projects onto the range of B.T
    project_kernel_B_matrix = np.identity(project_onto_range.shape[0]) - project_onto_range

    sim = np.dot(unfoldingInputData["R"], unfoldingInputData["sigma_0"])
    logu_0, _ = sinkhorn_iterations(
        sim, unfoldingInputData["nu"], cost_unbinned_normalized,
        prior_eps_reg, OT_prior_num_iterations, C_norm_constant,
    )

    logu_tilde = np.concatenate((logu_0, np.log(unfoldingInputData["sigma_0"])))

    logsigma, W2dist_OTiterations, OT_num_iterations = \
        generalized_sinkhorn_iterations_with_input_sigma_0(
            OT_num_iterations, b_vec, K_tilde, project_kernel_B_matrix,
            unfoldingInputData, logu_tilde, eps,
            cost_unbinned_normalized, cost_unbinned, W2_stopping,
        )
    normalizedsigma = np.exp(logsigma) / np.sum(np.exp(logsigma))
    return normalizedsigma, W2dist_OTiterations, OT_num_iterations


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def runIBUOT(unfoldingInputData, OT_num_iterations, W2_stopping,
             IBU_num_iterations, IBU_stopping, eps):
    """Run IBU and/or OT deconvolution, returning a dict of outputs."""

    # IBU requires y_measured and y_response to coincide (binned-grid setup)
    if len(unfoldingInputData["y_measured"]) != len(unfoldingInputData["y_response"]):
        print("Warning: y_measured and y_response are not the same length. IBU will not be run.")
        IBU_num_iterations = 0
    elif not (unfoldingInputData["y_measured"] == unfoldingInputData["y_response"]).all():
        print("Warning: y_measured and y_response are not the same. IBU will not be run.")
        IBU_num_iterations = 0

    od = {"IBU_num_iterations": IBU_num_iterations}

    # --- IBU ---
    if IBU_num_iterations > 0:
        t0_LR = time.time()
        od["sigma_IBU"], od["W2dist_IBUiterations"], od["IBU_num_iterations"] = \
            IBU(unfoldingInputData, IBU_num_iterations, IBU_stopping)
        od["T_LR"] = np.round(time.time() - t0_LR)
        od["noisy_sigma_IBU"] = np.dot(unfoldingInputData["R"], od["sigma_IBU"])
        od["noisy_sigma_IBU_unbinned"] = np.dot(unfoldingInputData["R_unbinned"], od["sigma_IBU"])
    else:
        od["sigma_IBU"] = None
        od["T_LR"] = None
        od["noisy_sigma_IBU"] = None
        od["noisy_sigma_IBU_unbinned"] = None
        od["W2dist_IBUiterations"] = None

    # --- OT ---
    if OT_num_iterations > 0:
        t0_OT = time.time()
        od["sigma_OT"], od["W2dist_OTiterations"], od["OT_num_iterations"] = \
            OT_Deconvolve(OT_num_iterations, W2_stopping, unfoldingInputData, eps)
        od["T_OT"] = np.round(time.time() - t0_OT)
        od["noisy_sigma_OT"] = np.dot(unfoldingInputData["R"], od["sigma_OT"])
        od["noisy_sigma_OT_unbinned"] = np.dot(unfoldingInputData["R_unbinned"], od["sigma_OT"])
    else:
        od["sigma_OT"] = None
        od["T_OT"] = None
        od["noisy_sigma_OT"] = None
        od["noisy_sigma_OT_unbinned"] = None
        od["W2dist_OTiterations"] = None
        od["OT_num_iterations"] = 0

    return od
