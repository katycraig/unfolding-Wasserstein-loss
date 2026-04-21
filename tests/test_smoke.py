#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal smoke tests: if this script runs without errors, the main code paths
of the repository are functioning.

Run with:

    python tests/test_smoke.py
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from OTDeconvolutionAlgorithm import runIBUOT           # noqa: E402
from ToyData1D_ImportData import setup1dtoydata         # noqa: E402
from MassData2D_ImportData import import2DMassDataBinAndUnbin  # noqa: E402


def test_1d_binned_ibu():
    d = setup1dtoydata(binNumber=20, samplingSeed=0)
    out = runIBUOT(d, OT_num_iterations=0, W2_stopping=1e-4,
                   IBU_num_iterations=5, IBU_stopping=1e-4, eps=3e-5)
    assert out["IBU_num_iterations"] == 5
    assert np.isclose(out["sigma_IBU"].sum(), 1.0, atol=1e-6)
    print("test_1d_binned_ibu: OK")


def test_1d_unbinned_ot():
    d = setup1dtoydata(binNumber=None, samplingSeed=0)
    out = runIBUOT(d, OT_num_iterations=5, W2_stopping=1e-4,
                   IBU_num_iterations=0, IBU_stopping=1e-4, eps=3e-5)
    assert out["OT_num_iterations"] == 5
    assert np.isclose(out["sigma_OT"].sum(), 1.0, atol=1e-6)
    print("test_1d_unbinned_ot: OK")


def test_2d_unbinned_ot():
    d = import2DMassDataBinAndUnbin(binNumber=None, n_prior=50, n_sigma_true=50)
    out = runIBUOT(d, OT_num_iterations=3, W2_stopping=1e-4,
                   IBU_num_iterations=0, IBU_stopping=1e-4, eps=3e-5)
    assert out["OT_num_iterations"] == 3
    assert np.isclose(out["sigma_OT"].sum(), 1.0, atol=1e-6)
    print("test_2d_unbinned_ot: OK")


def test_2d_binned_ibu():
    d = import2DMassDataBinAndUnbin(binNumber=10, n_prior=50, n_sigma_true=50)
    out = runIBUOT(d, OT_num_iterations=0, W2_stopping=1e-4,
                   IBU_num_iterations=3, IBU_stopping=1e-4, eps=3e-5)
    assert out["IBU_num_iterations"] == 3
    assert np.isclose(out["sigma_IBU"].sum(), 1.0, atol=1e-6)
    print("test_2d_binned_ibu: OK")


if __name__ == "__main__":
    test_1d_binned_ibu()
    test_1d_unbinned_ot()
    test_2d_unbinned_ot()
    test_2d_binned_ibu()
    print("\nAll smoke tests passed.")
