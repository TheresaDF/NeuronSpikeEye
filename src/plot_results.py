from matplotlib import pyplot as plt 
import seaborn as sns 
import numpy as np
import glob 
import pickle
import os  
import pandas as pd

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
    
    data_list = []

    for c, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        snr_idx = c // n_repeats
        snr_value = snrs[snr_idx]

        channel_true = np.array([np.sum(data['true'][i, :]) for i in range(32)])

        for method in ['Mean', 'Threshold', 'Wavelet', 'SVR']:
            try:
                if method == 'Mean': 
                    channel_est = np.mean(data['estimated_mean_predict'])
                elif method == 'Threshold':
                    channel_est = np.array([np.sum(data['estimated_baseline'][i]) for i in range(32)])
                elif method == 'Wavelet':
                    channel_est = np.array([np.sum(data['estimated_wavelet'][i]) for i in range(32)])
                elif method == 'SVR':
                    channel_est = data['estimated_svm']
                
                error = channel_est - channel_true
                for val in error:
                    data_list.append({"SNR": snr_value, "Error": val, "Method": method})
            except:
                pass
    
    df = pd.DataFrame(data_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="SNR", y="Error", hue="Method", split=False, inner="quart", palette=[[0.99, 0.46, 0.2 ], [0.969, 0.733, 0.694], [0.91 , 0.247, 0.282], [0.475, 0.137, 0.557]], ax=ax)
    
    ax.set_xlabel(r"$\alpha$ Levels")
    ax.set_ylabel("Difference (Estimation - True)")
    ax.set_title("Comparison of Error for Different Methods")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def generate_plots(path, data_type, stim, n_repeats: int = 5, snrs: list = [0.1, 0.5, 1, 1.5, 2]):
    os.makedirs("results", exist_ok=True)
    noise_dist = os.listdir(path)
    noise_dist = [dist for dist in noise_dist if "noise" in dist]
    for noise in noise_dist:
        fig = plot_results(path + noise, n_repeats=n_repeats, snrs=snrs)
        name = f"{data_type}_{stim}_{noise}"
        savefig(fig, name, width=5.5, height=3)


if __name__ == "__main__": 
    data_type = "synthetic"
    stim = "stim"

    # path = f"results/results_" + data_type + "_" + stim + "/"
    path = f"../../../../../../work3/s194329/results_" + data_type + "_" + stim + "/"
    generate_plots(path, data_type, stim, n_repeats=30, snrs=np.r_[0.1, np.arange(1, 7)])
