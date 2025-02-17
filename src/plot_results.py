from matplotlib import pyplot as plt 
from scipy.stats import t
import seaborn as sns 
import numpy as np
import glob 
import pickle
import os  

sns.set_theme()


rcParams = {
    "font.family": "serif",  # use serif/main font for text elements
    'text.usetex': True,
    'font.size': 8,
    'axes.labelsize': 7,
    'axes.titlesize': 9,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'axes.labelpad': 1,
    'axes.axisbelow': True,  # draw gridlines below other elements
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([
        r"\usepackage{url}",            # load additional packages
        r"\usepackage{amsmath,amssymb}",   # unicode math setup
        #  r"\setmainfont{DejaVu Serif}",  # serif font via preamble
    ])
}
plt.rcParams.update(rcParams)

figdir = "results/"
def savefig(fig, name, width=6, height=3):
    # apply rcParams
    fig.set_size_inches(width, height)
    fig.savefig(figdir + name + ".pdf", bbox_inches='tight')



def plot_results(path: str, snrs: list = [0.1, 0.5, 1, 1.5, 2], n_repeats: int = 5):
    files = sorted(glob.glob(path + '/*.pkl'), key=len)

    num_snrs = len(snrs)
    errors_baseline = np.zeros((num_snrs, n_repeats)); errors_baseline[:] = np.nan 
    errors_wavelet = np.zeros((num_snrs, n_repeats));  errors_wavelet[:] = np.nan 
    errors_svm = np.zeros((num_snrs, n_repeats));      errors_svm[:] = np.nan 

    for c, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        snr_idx = c // n_repeats
        repeat_idx = c % n_repeats

        channel_true = [np.sum(data['true'][i, :]) for i in range(32)]
        # Compute relative errors as percentages
        try:
            channel_base = np.array([np.sum(data['estimated_baseline'][i]) for i in range(32)]) 
            errors_baseline[snr_idx, repeat_idx] = np.sqrt(np.mean((channel_base - channel_true)**2))
        except:
            pass  
        try: 
            channel_wave = np.array([np.sum(data['estimated_wavelet'][i]) for i in range(32)])
            errors_wavelet[snr_idx, repeat_idx] = np.sqrt(np.mean((channel_wave - channel_true)**2))
        except: 
            pass 
        try: 
            channel_svr = data['estimated_svm']
            errors_svm[snr_idx, repeat_idx] = np.sqrt(np.mean((channel_svr - channel_true)**2))
        except: 
            pass 

    # Calculate mean, std, and confidence intervals for each method
    def calculate_stats(errors):
        means = np.nanmean(errors, axis=1)
        stds = np.nanstd(errors, axis=1, ddof=1)
        n = errors.shape[1]
        t_value = t.ppf(0.975, df=n-1)  # 95% confidence, n-1 degrees of freedom
        cis = t_value * (stds / np.sqrt(n))
        return means, cis

    baseline_mean, baseline_ci = calculate_stats(errors_baseline)
    wavelet_mean, wavelet_ci = calculate_stats(errors_wavelet)
    svm_mean, svm_ci = calculate_stats(errors_svm)

    # Plot scatter with error bars (confidence intervals)
    x = np.arange(len(snrs))  # the label locations
    fig, ax = plt.subplots(figsize=(10, 6))

    offset = 0.1  # Offset for separating dots

    ax.errorbar(x - offset, baseline_mean, yerr=baseline_ci, fmt='o', label='Baseline', color='darkblue', capsize=5)
    ax.errorbar(x, wavelet_mean, yerr=wavelet_ci, fmt='o', label='Wavelet', color='pink', capsize=5)
    ax.errorbar(x + offset, svm_mean, yerr=svm_ci, fmt='o', label='SVM', color='darkred', capsize=5)

    # Add labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels([f"{snr}" for snr in snrs])
    ax.set_xlabel(r"$\alpha$ Levels")
    ax.set_ylabel(r"RMSE")
    ax.set_title("Comparison of Error for Different Methods")
    ax.set_ylim([0, 90])
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig

def generate_plots(path, n_repeats : int = 5, snrs : list = [0.1, 0.5, 1, 1.5, 2]):
    os.makedirs("results", exist_ok = True)
    noise_dist = os.listdir(path)
    noise_dist = [dist for dist in noise_dist if "noise" in dist]
    for noise in noise_dist:
        fig = plot_results(path + noise, n_repeats = n_repeats, snrs = snrs)
        name = "noise_" + noise
        savefig(fig, name)

if __name__ == "__main__": 
    path = f"../../../../../../work3/s194329/results/"
    generate_plots(path, n_repeats = 30, snrs = np.r_[0.1, np.arange(1, 11)])
