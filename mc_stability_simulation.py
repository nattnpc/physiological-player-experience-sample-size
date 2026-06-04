"""
Monte Carlo Stability Simulation for Affordance-Level Physiological Player Experience Research
================================================================================================

Supplementary material for:
    Rein, N., & Günter, T. (in press). How Many Players Do You Need?
    A Monte Carlo Framework for Sample Size Planning in Physiological
    Player Experience Research. Behavior Research Methods.

Description
-----------
This script implements the Monte Carlo stability simulation and convergent
resampling analysis described in the paper. Given a player × affordance
EDA matrix, it computes:
  1. Monte Carlo stability curves: how Spearman rank correlation between
     subsample and full-sample coefficient orderings varies with N.
  2. Convergent resampling curves: how mean pairwise cosine profile
     similarity varies with N.
  3. Practical guidance: at what N do estimates stabilize?

Usage
-----
1. Prepare your data as a pandas DataFrame:
   - Rows: players (one row per player)
   - Columns: affordances (one column per gameplay affordance)
   - Values: mean normalized EDA per player per affordance (float)
   - Missing values (affordance not encountered): NaN or 0 (both handled)

2. Adjust the parameters in the CONFIG section below.

3. Run: python mc_stability_simulation.py

Output
------
  - mc_stability.png    : Monte Carlo curves (mean ρ with CI ribbon)
  - mc_pct_above.png    : % of samples above ρ threshold
  - conv_resampling.png : Convergent resampling curves
  - mc_results.csv      : Full numerical results table
  - A printed summary with practical recommendations

Requirements
------------
  Python 3.8+
  pandas, numpy, scikit-learn, scipy, matplotlib

Install: pip install pandas numpy scikit-learn scipy matplotlib

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — adjust these parameters for your study
# ============================================================

# Path to your data file
# Your data should be a CSV or Excel with players as rows, affordances as columns
DATA_FILE   = "your_data.csv"   # or .xlsx
SHEET_NAME  = None               # set to sheet name if Excel, else None for CSV
INDEX_COL   = 0                  # column index to use as player ID (0 = first column)

# Alternatively, load your DataFrame directly in code below and set DATA_FILE = None

# Ridge regression parameter
LAMBDA = 1.0          # regularization strength (sensitivity tested across 0.1–10.0)

# Monte Carlo parameters
N_ITER_MC   = 1000    # iterations per N (1000 recommended; use 500 for speed)
RHO_THRESHOLD = 0.70  # stability threshold for rank correlation

# Convergent resampling parameters
N_ITER_CONV = 100     # iterations per N (100 recommended)
DELTA_THRESHOLD = 0.005  # incremental gain below which similarity is considered stable

# Output
GAME_NAME   = "My Game"          # used in figure titles
SAVE_FIGURES = True              # set to False to display only
FIG_DPI     = 300

# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_data(filepath, sheet_name=None, index_col=0):
    """Load player x affordance EDA matrix from CSV or Excel."""
    if filepath is None:
        raise ValueError("DATA_FILE is None — load your DataFrame directly in the script.")
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=index_col)
    else:
        df = pd.read_csv(filepath, index_col=index_col)
    df = df.replace(0, np.nan)
    return df


def ridge_coef_full(df, lambda_val=1.0):
    """
    Fit ridge regression on the full dataset and return coefficient vector.
    y = row-wise mean of affordance EDA values (player's overall arousal profile).
    """
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(df.values.astype(float))
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = np.mean(X, axis=1)
    return Ridge(alpha=lambda_val).fit(Xs, y).coef_


def profile_sim_lopo(df):
    """
    Compute mean pairwise cosine similarity between LOPO-fold coefficient vectors.
    Returns mean similarity across all player pairs.
    """
    imp = SimpleImputer(strategy='mean')
    X = imp.fit_transform(df.values.astype(float))
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    y = np.mean(X, axis=1)
    n = len(X)
    coefs = []
    for i in range(n):
        train = [j for j in range(n) if j != i]
        coefs.append(Ridge(alpha=1.0).fit(Xs[train], y[train]).coef_)
    cm = np.array(coefs)
    sims = [1 - cosine(cm[i], cm[j])
            for i in range(n) for j in range(i+1, n)]
    return np.mean(sims)


def run_monte_carlo(df, n_iter=1000, lambda_val=1.0, rho_threshold=0.70, seed=42):
    """
    Monte Carlo stability simulation.

    For each N from 3 to len(df), draws n_iter random subsamples without
    replacement and computes Spearman rank correlation between subsample
    and full-sample coefficient orderings.

    Returns a dict with keys:
      n_vals      : list of N values tested
      means       : mean Spearman ρ at each N
      sds         : SD of Spearman ρ at each N
      ci_low      : 2.5th percentile at each N
      ci_high     : 97.5th percentile at each N
      pct_above   : % of samples with ρ > rho_threshold at each N
      all_rhos    : list of arrays, one per N (all iteration rhos)
    """
    np.random.seed(seed)
    n_max = len(df)
    n_aff = df.shape[1]
    full_coef  = ridge_coef_full(df, lambda_val)
    full_order = np.argsort(full_coef)[::-1]

    n_vals, means, sds, ci_low, ci_high, pct_above, all_rhos = [], [], [], [], [], [], []

    print(f"Running Monte Carlo for {GAME_NAME} (N_max={n_max}, {n_aff} affordances)...")
    for n_val in range(3, n_max + 1):
        rhos = []
        for _ in range(n_iter):
            idx  = np.random.choice(n_max, size=n_val, replace=False)
            samp = df.iloc[idx]
            try:
                c = ridge_coef_full(samp, lambda_val)
                samp_order = np.argsort(c)[::-1]
                fr = [int(np.where(full_order==i)[0][0])+1 for i in range(n_aff)]
                sr = [int(np.where(samp_order==i)[0][0])+1 for i in range(n_aff)]
                rho, _ = spearmanr(fr, sr)
                rhos.append(rho)
            except:
                pass
        rhos = np.array(rhos)
        n_vals.append(n_val)
        means.append(rhos.mean())
        sds.append(rhos.std())
        ci_low.append(np.percentile(rhos, 2.5))
        ci_high.append(np.percentile(rhos, 97.5))
        pct_above.append(np.mean(rhos > rho_threshold) * 100)
        all_rhos.append(rhos)
        print(f"  N={n_val:2d}: mean ρ={rhos.mean():.3f}, "
              f"95%CI=[{np.percentile(rhos,2.5):.3f},{np.percentile(rhos,97.5):.3f}], "
              f"% ρ>{rho_threshold}={np.mean(rhos>rho_threshold)*100:.1f}%")

    return dict(n_vals=n_vals, means=means, sds=sds,
                ci_low=ci_low, ci_high=ci_high,
                pct_above=pct_above, all_rhos=all_rhos)


def run_convergent_resampling(df, n_iter=100, seed=42, delta=0.005):
    """
    Convergent resampling analysis.

    For each N from 3 to len(df), draws n_iter random subsamples and
    computes mean pairwise cosine profile similarity for each.

    Returns a dict with keys:
      n_vals    : list of N values tested
      means     : mean profile similarity at each N
      ci_low    : 2.5th percentile
      ci_high   : 97.5th percentile
      stab_n    : first N where incremental gain < delta (None if not found)
    """
    np.random.seed(seed)
    n_max = len(df)
    n_vals, means, ci_low, ci_high = [], [], [], []

    print(f"\nRunning convergent resampling for {GAME_NAME}...")
    for n_val in range(3, n_max + 1):
        sims = []
        for _ in range(n_iter):
            idx  = np.random.choice(n_max, size=n_val, replace=False)
            samp = df.iloc[idx]
            try:
                sims.append(profile_sim_lopo(samp))
            except:
                pass
        n_vals.append(n_val)
        means.append(np.mean(sims))
        ci_low.append(np.percentile(sims, 2.5))
        ci_high.append(np.percentile(sims, 97.5))
        delta_val = means[-1] - means[-2] if len(means) > 1 else 0
        stab_str = ' ← Δ<0.005 (stable)' if abs(delta_val) < delta and n_val > 3 else ''
        print(f"  N={n_val:2d}: M={means[-1]:.3f}, "
              f"95%CI=[{ci_low[-1]:.3f},{ci_high[-1]:.3f}]{stab_str}")

    # Find stabilization point
    means_arr = np.array(means)
    stab_n = None
    for i in range(1, len(means_arr)):
        if abs(means_arr[i] - means_arr[i-1]) < delta:
            stab_n = n_vals[i]
            break

    return dict(n_vals=n_vals, means=means, ci_low=ci_low,
                ci_high=ci_high, stab_n=stab_n)


def plot_monte_carlo(mc, game_name, rho_threshold=0.70, save=True):
    """Plot Monte Carlo stability curve with CI ribbon."""
    n_vals  = np.array(mc['n_vals'])
    means   = np.array(mc['means'])
    ci_low  = np.array(mc['ci_low'])
    ci_high = np.array(mc['ci_high'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(n_vals, ci_low, ci_high, alpha=0.2, color='#7B5EA7',
                    label='95% CI')
    ax.plot(n_vals, means, 'o-', color='#4A3570', lw=2, markersize=6,
            label='Mean rank correlation (ρ)')
    ax.axhline(rho_threshold, color='#CC4444', lw=1.5, ls='--',
               label=f'ρ = {rho_threshold} threshold')

    # Mark first N where mean > threshold
    above = [n for n, m in zip(n_vals, means) if m > rho_threshold]
    if above:
        ax.axvline(above[0], color='#CC4444', lw=1.0, ls=':', alpha=0.8)
        ax.text(above[0] + 0.15, 0.05, f'N={above[0]}',
                fontsize=9, color='#CC4444', fontweight='bold')

    # Annotate each point
    for n, m in zip(n_vals, means):
        ax.annotate(f'{m:.2f}', (n, m),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=7.5, color='#4A3570')

    ax.set_xlabel('Sample Size (N)', fontsize=11)
    ax.set_ylabel('Mean Spearman ρ with Full-Sample Ordering', fontsize=11)
    ax.set_title(f'Monte Carlo Coefficient Stability — {game_name}\n'
                 f'({len(mc["all_rhos"][0])} iterations per N)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim([n_vals[0] - 0.5, n_vals[-1] + 0.5])
    ax.set_ylim([-0.15, 1.15])
    ax.set_xticks(n_vals)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig('mc_stability.png', bbox_inches='tight', dpi=FIG_DPI)
        print('Saved: mc_stability.png')
    plt.show()


def plot_pct_above(mc, game_name, rho_threshold=0.70, save=True):
    """Plot % of samples achieving rho > threshold."""
    n_vals   = np.array(mc['n_vals'])
    pct_above = np.array(mc['pct_above'])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(n_vals, pct_above, color='#7B5EA7', alpha=0.85, width=0.6)
    ax.axhline(70, color='#CC4444', lw=1.5, ls='--', label='70% threshold')

    for bar, pct in zip(bars, pct_above):
        if pct > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{pct:.0f}%', ha='center', fontsize=8, color='#333333')

    ax.set_xlabel('Sample Size (N)', fontsize=11)
    ax.set_ylabel(f'% Samples with ρ > {rho_threshold}', fontsize=11)
    ax.set_title(f'Proportion of Stable Subsamples — {game_name}',
                 fontsize=11, fontweight='bold')
    ax.set_xlim([n_vals[0] - 0.5, n_vals[-1] + 0.5])
    ax.set_ylim([0, 115])
    ax.set_xticks(n_vals)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig('mc_pct_above.png', bbox_inches='tight', dpi=FIG_DPI)
        print('Saved: mc_pct_above.png')
    plt.show()


def plot_convergent_resampling(conv, game_name, save=True):
    """Plot convergent resampling curve."""
    n_vals  = np.array(conv['n_vals'])
    means   = np.array(conv['means'])
    ci_low  = np.array(conv['ci_low'])
    ci_high = np.array(conv['ci_high'])
    stab_n  = conv['stab_n']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(n_vals, ci_low, ci_high, alpha=0.2, color='#7B5EA7',
                    label='95% CI')
    ax.plot(n_vals, means, 'o-', color='#4A3570', lw=2, markersize=6,
            label='Mean profile similarity')

    if stab_n:
        ax.axvline(stab_n, color='#CC4444', lw=1.5, ls='--',
                   label=f'N={stab_n}: Δ<0.005 (stable)')
        ax.axvspan(stab_n, n_vals[-1] + 0.4, alpha=0.07, color='green',
                   label='Stable region')

    for n, m in zip(n_vals, means):
        ax.annotate(f'{m:.3f}', (n, m),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=7.5, color='#4A3570')

    ax.set_xlabel('Sample Size (N)', fontsize=11)
    ax.set_ylabel('Mean Pairwise Cosine Profile Similarity', fontsize=11)
    ax.set_title(f'Convergent Resampling Curve — {game_name}',
                 fontsize=11, fontweight='bold')
    ax.set_xlim([n_vals[0] - 0.5, n_vals[-1] + 0.5])
    ax.set_ylim([0.3, 1.05])
    ax.set_xticks(n_vals)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig('conv_resampling.png', bbox_inches='tight', dpi=FIG_DPI)
        print('Saved: conv_resampling.png')
    plt.show()


def save_results_csv(mc, conv, game_name):
    """Save full numerical results to CSV."""
    n_vals = mc['n_vals']
    rows = []
    for i, n_val in enumerate(n_vals):
        rows.append({
            'N': n_val,
            'MC_mean_rho': mc['means'][i],
            'MC_sd_rho':   mc['sds'][i],
            'MC_ci_low':   mc['ci_low'][i],
            'MC_ci_high':  mc['ci_high'][i],
            'MC_pct_above_threshold': mc['pct_above'][i],
            'Conv_mean_similarity': conv['means'][i] if i < len(conv['means']) else None,
            'Conv_ci_low': conv['ci_low'][i] if i < len(conv['ci_low']) else None,
            'Conv_ci_high': conv['ci_high'][i] if i < len(conv['ci_high']) else None,
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv('mc_results.csv', index=False)
    print('Saved: mc_results.csv')


def print_summary(mc, conv, game_name, rho_threshold=0.70):
    """Print a practical summary of results."""
    n_vals = mc['n_vals']
    means  = mc['means']
    pct    = mc['pct_above']
    stab_n = conv['stab_n']

    print(f"\n{'='*65}")
    print(f"  STABILITY SUMMARY — {game_name}")
    print(f"{'='*65}")
    print(f"  N_max = {max(n_vals)}, Affordances = {mc['all_rhos'][0].shape[0] if hasattr(mc['all_rhos'][0], 'shape') else 'unknown'}")
    print(f"  Monte Carlo: {N_ITER_MC} iterations per N")
    print(f"  Stability threshold: ρ > {rho_threshold}")
    print()

    # Find key N values
    above70 = [n for n, m in zip(n_vals, means) if m > 0.70]
    above80 = [n for n, m in zip(n_vals, means) if m > 0.80]
    pct70   = [n for n, p in zip(n_vals, pct) if p >= 70]

    print(f"  Mean ρ first exceeds 0.70 at N = {above70[0] if above70 else 'not reached'}")
    print(f"  Mean ρ first exceeds 0.80 at N = {above80[0] if above80 else 'not reached'}")
    print(f"  ≥70% of samples above threshold at N = {pct70[0] if pct70 else 'not reached'}")
    print(f"  Profile similarity stabilizes (Δ<0.005) at N = {stab_n if stab_n else 'not detectable (N_max too small)'}")
    print()
    print(f"  Practical recommendation:")
    if above70:
        rec_n = above70[0]
        if rec_n <= 8:
            tier = "MINIMUM (use only for pilot/proof-of-concept work)"
        elif rec_n <= 14:
            tier = "RECOMMENDED for primary empirical studies"
        else:
            tier = "ROBUST — required for this game's affordance structure"
        print(f"    Target N ≥ {rec_n} ({tier})")
    else:
        print(f"    Stability not achieved within available N — collect more data")
    print(f"{'='*65}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # ── Load data ────────────────────────────────────────────────────────
    # Option A: Load from file
    if DATA_FILE is not None:
        df = load_data(DATA_FILE, sheet_name=SHEET_NAME, index_col=INDEX_COL)
    else:
        # Option B: Construct your DataFrame directly here
        # Example:
        # df = pd.DataFrame({
        #     'Affordance_A': [0.45, 0.62, 0.38, ...],
        #     'Affordance_B': [0.71, 0.55, 0.80, ...],
        # }, index=['P01', 'P02', 'P03', ...])
        raise ValueError("Set DATA_FILE or construct df directly in the script.")

    print(f"Data loaded: {df.shape[0]} players, {df.shape[1]} affordances")
    print(f"Missing values per affordance:")
    print(df.isnull().sum())

    # ── Run analyses ─────────────────────────────────────────────────────
    mc_results   = run_monte_carlo(df, n_iter=N_ITER_MC,
                                   lambda_val=LAMBDA,
                                   rho_threshold=RHO_THRESHOLD)

    conv_results = run_convergent_resampling(df, n_iter=N_ITER_CONV,
                                             delta=DELTA_THRESHOLD)

    # ── Plot figures ─────────────────────────────────────────────────────
    plot_monte_carlo(mc_results, GAME_NAME,
                     rho_threshold=RHO_THRESHOLD,
                     save=SAVE_FIGURES)

    plot_pct_above(mc_results, GAME_NAME,
                   rho_threshold=RHO_THRESHOLD,
                   save=SAVE_FIGURES)

    plot_convergent_resampling(conv_results, GAME_NAME,
                               save=SAVE_FIGURES)

    # ── Save results and print summary ───────────────────────────────────
    save_results_csv(mc_results, conv_results, GAME_NAME)
    print_summary(mc_results, conv_results, GAME_NAME,
                  rho_threshold=RHO_THRESHOLD)
