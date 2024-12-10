from matplotlib import pyplot as plt 
import numpy as np
from scipy.fft import fft, fftfreq, ifft 
from sklearn.decomposition import FastICA  
from scipy.stats import entropy
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import acf

def time_to_freq(data, sample_rate = 3*1e4):
    """
    Function to convert time domain data to frequency domain data
    """
    duration = len(data) / sample_rate
    N  = int(sample_rate * duration)
    yf = fft(data)
    xf = fftfreq(N, 1 / sample_rate)

    return xf, yf 

def freq_to_time(yf):
    """
    Function to convert frequency domain data to time domain data
    """
    ifft_data = ifft(yf)
    return ifft_data


def perform_ICA(data, n_components = 32):
    """
    Function to perform ICA on the data
    """
    ica = FastICA(n_components = n_components, whiten_solver = "svd", max_iter = 1000)
    S_ = ica.fit_transform(data)
    return ica, S_

def remove_ica_components(ica, data, idx):
    """
    Function to remove ICA components
    """
    modified_data = np.copy(data)
    modified_data[:, idx] = 0 
    return ica.inverse_transform(modified_data)

def remove_drift(data, freq = 1):
    """
    Function to remove drift from the data
    """
    bins = np.copy(data)
    for i in range(bins.shape[1]):
        xf, yf = time_to_freq(bins[:, i])
        yf[np.abs(xf) < freq] = 0 
        bin_ = np.real(freq_to_time(yf))
        bins[:, i] = bin_

    return bins 


def compute_num_components(data, threshold = 0.95, to_plot = False) -> np.ndarray:
    """
    Function to compute the number of components to keep
    """
   
    n_comp = np.arange(1, data.shape[1]+1)
    loss = np.zeros(len(n_comp))
    for i, n in enumerate(n_comp):
        ica, data_trans = perform_ICA(data, n_components = n)

        # compute reconstruction loss 
        loss_current = np.linalg.norm(data - ica.inverse_transform(data_trans))
        loss[i] = loss_current

    loss = np.cumsum(loss)
    loss = loss / loss[-1]

    # compute how many components to keep
    idx = np.where(loss < threshold)[0][-1] + 1

    if to_plot:
        plt.figure(figsize = (10, 5))
        plt.plot(n_comp, loss, color = "darkblue", marker = "o")
        plt.plot(np.arange(-1, data.shape[1]+2), np.ones(len(n_comp)+3) * threshold, '--', color = "gray")
        plt.xlabel("Number of components")
        plt.ylabel("Reconstruction loss")
        plt.xlim([0, data.shape[1]])
        plt.show()

    return idx


def find_bad_channels(data : np.ndarray, n_comp : int) -> np.ndarray:
    """ Function to find bad channels using ICA"""

    acfs_all = np.zeros(n_comp)
    for i in range(n_comp):
        acfs_all[i] = acf(data[:, i], nlags=20*30)[-1]
    
    bad_component = np.max(np.abs(acfs_all))
    return bad_component 


def butter_lowpass(data : np.ndarray, cutoff : int = 5000, fs : int = 30000, order=10):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)



def filter(data : np.ndarray) -> np.ndarray:
    """ Use ICA to filtef data"""

    # remove drift 
    data = remove_drift(data)
    data = butter_lowpass(data)

    # compute number of components to keep 
    n_comp = max(7, compute_num_components(data, threshold = 0.95))

    # perform ICA
    ica, data_trans = perform_ICA(data, n_components = n_comp)

    # remove bad components
    idx = find_bad_channels(data_trans, n_comp)
    data_filtered = remove_ica_components(ica, data_trans, idx)
    
    return data_filtered