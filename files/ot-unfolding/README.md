# OT Unfolding

Optimal-transport (OT) deconvolution and Iterative Bayesian Unfolding (IBU / Richardson–Lucy) for unfolding / deconvolving experimental distributions. Includes a 1D toy-data demo and a 2D demo on jet-mass data.

## Repository layout

```
ot-unfolding/
├── data/                       # input data files
│   ├── jet_mass.txt
│   └── jet_mass_SD.txt
├── src/
│   ├── OTDeconvolutionAlgorithm.py     # OT deconvolution + IBU algorithms
│   ├── ToyData1D_ImportData.py         # 1D toy-data generator
│   ├── MassData2D_ImportData.py        # 2D jet-mass data loader
│   └── PlottingFunctions.py            # plotting utilities
├── examples/
│   ├── ManySeeds.py                    # 1D toy-data comparison over many seeds
│   └── MassData2D_Examples.py          # 2D jet-mass comparison over bin counts
├── tests/
│   └── test_smoke.py                   # quick sanity checks
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/ot-unfolding.git
cd ot-unfolding
python -m venv .venv && source .venv/bin/activate       # optional
pip install -r requirements.txt
```

## Running the examples

From the repository root:

```bash
# 1D toy data across multiple seeds and bin counts
python examples/ManySeeds.py

# 2D jet-mass data across different binnings
python examples/MassData2D_Examples.py
```

Each example writes a pickle (`temp1ddata.pkl` or `tempdata.pkl`) and renders summary plots.

## Algorithms

- **OT deconvolution** is a generalized Sinkhorn / Douglas–Rachford algorithm that solves an optimal-transport formulation of the deconvolution problem directly on the (unbinned) measured data.
- **Iterative Bayesian Unfolding** (equivalently Richardson–Lucy) is implemented as a reference / comparison method. IBU requires the data to be binned so that the measured and simulated distributions share the same support.

In the example scripts, OT is run on unbinned data while IBU is run across several bin counts to illustrate the dependence of binned methods on the binning choice.

## Configuration

The following environment variables are recognized:

| Variable | Effect |
|---|---|
| `OT_UNFOLDING_DATA_DIR` | Directory containing `jet_mass.txt` and `jet_mass_SD.txt`. Defaults to `data/` next to this repo. |
| `OT_UNFOLDING_USE_LATEX` | Set to `1` to render figure text with LaTeX (requires a local `latex` + `dvipng` install). Default is `0`, which falls back to matplotlib's mathtext. |
| `MPLBACKEND` | Standard matplotlib backend override (e.g. `Agg` for headless servers). |

## Citation

If you use this code in a scientific publication, please cite the accompanying paper (link TBD) and this repository.

## License

This project is released under the [MIT License](LICENSE).
