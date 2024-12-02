from scipy.signal import find_peaks
from matplotlib import pyplot as plt 
import numpy as np
from scipy.fft import fft, fftfreq, ifft 
from sklearn.decomposition import FastICA  
from scipy.signal import iirnotch, lfilter
from sklearn.cluster import KMeans

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

def remove_drift(bins_, freq = 1):
    """
    Function to remove drift from the data
    """
    bins = np.copy(bins_)
    for i in range(100):
        xf, yf = time_to_freq(bins[i])
        yf[np.abs(xf) < freq] = 0 
        bin_ = freq_to_time(yf)
        bins[i] = bin_

    return bins 


def compute_num_components(data, threshold = 0.95, to_plot = False):
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
    val = []
    for i in range(n_comp):
        IQR = np.percentile(data[:, i], 75) - np.percentile(data[:, i], 25)   
        val.append(IQR)

    # normalize and fit kmeans 
    val = np.array(val)
    val = val / np.max(val)
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(np.c_[val, np.zeros(len(val))])
    labels = kmeans.labels_

    # find the indices from the class that is smallest (it may change between runs)
    idx = np.where(labels == 0)[0] if len(np.where(labels == 0)[0]) < len(np.where(labels == 1)[0]) else np.where(labels == 1)[0]

    return idx


def filter(data : np.ndarray) -> np.ndarray:
    """ Use ICA to filtef data"""

    # compute number of components to keep 
    n_comp = max(7, compute_num_components(data, threshold = 0.95))

    # perform ICA
    ica, data_trans = perform_ICA(data, n_components = n_comp)

    # remove bad components
    idx = find_bad_channels(data_trans, n_comp)
    data_filtered = remove_ica_components(ica, data_trans, idx)

    # remove drift
    data_filtered = remove_drift(data_filtered)
    
    return data_filtered