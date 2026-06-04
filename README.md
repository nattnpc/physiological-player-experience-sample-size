# Monte Carlo Stability Simulation for Physiological Player Experience Research

Supplementary code for:

> How Many Players Do You Need? 
> A Monte Carlo Framework for Sample Size Planning in Physiological 
> Player Experience Research. *Behavior Research Methods*.

## Contents

- `mc_stability_simulation.py` — Documented Python script. 
  Accepts any player × affordance EDA matrix as input.
- `mc_methods_notebook.ipynb` — Jupyter notebook reproducing 
  all analyses across three game datasets.

## Requirements

Python 3.8+, pandas, numpy, scikit-learn, scipy, matplotlib

Install: `pip install pandas numpy scikit-learn scipy matplotlib`

## Usage

Edit the CONFIG section in `mc_stability_simulation.py` to point 
to your data file, then run: `python mc_stability_simulation.py`

## Data

The anonymized EDA datasets used in the paper are available from 
the corresponding author upon reasonable request.

## License

MIT License — free to use, adapt, and build upon with attribution.
