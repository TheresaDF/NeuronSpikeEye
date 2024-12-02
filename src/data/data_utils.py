from scipy.signal import find_peaks
import numpy as np
from scipy.fft import fft, fftfreq, ifft 
from sklearn.decomposition import FastICA, PCA 
from scipy.signal import iirnotch, lfilter
from sklearn.cluster import KMeans


def time_to_freq(data : np.ndarray, sample_rate : int = 3*1e4) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to convert time domain data to frequency domain data
    """
    duration = len(data) / sample_rate
    N  = int(sample_rate * duration)
    yf = fft(data)
    xf = fftfreq(N, 1 / sample_rate)

    return xf, yf 

def freq_to_time(yf : np.ndarray) -> np.ndarray:
    """
    Function to convert frequency domain data to time domain data
    """
    ifft_data = ifft(yf)
    return ifft_data



def high_pass(data : np.ndarray, threshold : int = 1) -> np.ndarray: 
    """Function to apply high pass filter"""
    
    for i in range(data.shape[0]):
        xf, yf = time_to_freq(data[i])
        yf[xf < threshold] = 0

        data[i] = freq_to_time(yf)
    
    return data 

def low_pass(data : np.ndarray, threshold : int = 10000) -> np.ndarray: 
    """Function to apply high pass filter"""
    
    for i in range(data.shape[0]):
        xf, yf = time_to_freq(data[i])
        yf[xf > threshold] = 0

        data[i] = freq_to_time(yf)
    
    return data 

def notch_filter(x : np.ndarray, f0 : float, fs : float, Q : int = 30) -> np.ndarray:
    b, a = iirnotch(f0 / (fs / 2), Q)
    y = lfilter(b, a, x)
    
    return y

def apply_notch_filter(data : np.ndarray, f0 : float, num : int, sample_rate : float = 3*1e4, Q : int = 30) -> np.ndarray:
    """
    Apply notch filter to data
    """
    filtered_data = data.copy()
    for i in range(num): 
        filtered_data = notch_filter(filtered_data, (i+1) * f0, sample_rate, Q = Q)

    return filtered_data

def remove_ica_components(ica : FastICA, data : np.ndarray, idx : int) -> np.ndarray:
    """
    Function to remove ICA components
    """
    modified_data = np.copy(data)
    modified_data[:, idx] = 0 
    return ica.inverse_transform(modified_data)


def find_num_components(data : np.ndarray, threshold : float = 0.975) -> int:
    """ Function to find the number of components to keep in PCA/ICA"""
    pca = PCA(n_components = 32)
    pca.fit(data)
    num = np.sum(np.cumsum(pca.explained_variance_ratio_) < threshold)

    return num

def find_bad_channels_ica(data_trans : np.ndarray, n_comp : int) -> np.ndarray:
    """
    Function to find the peak channels in the ICA data
    """
    val = np.zeros(n_comp)
    for i in range(n_comp):
        IQR = np.percentile(data_trans[:, i], 75) - np.percentile(data_trans[:, i], 25)   
        val[i] = IQR

    # Apply KMeans clustering to find the peak channels
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(np.c_[val, np.zeros(len(val))])
    labels = kmeans.labels_

    return np.where(labels == 1)[0]

def perform_ICA(data, n_components = 32):
    """
    Function to perform ICA on the data
    """
    ica = FastICA(n_components = n_components, whiten_solver = "eigh", max_iter = 1000)
    S_ = ica.fit_transform(data)
    return ica, S_

def filter(data : np.ndarray, apply_notch : bool = False) -> np.ndarray:
    """
    Function to filter data
    """
    
    num_components = max(find_num_components(data), 9)
    ica, S_ = perform_ICA(data, n_components = num_components)
    bad_channels = find_bad_channels_ica(S_, num_components)
    filtered_data = remove_ica_components(ica, S_, bad_channels)

    if apply_notch:
    # apply notch filters to filter away the 50hz and 500hz noise
        filtered_data = apply_notch_filter(filtered_data, 50, 5, Q = 3)
        filtered_data = apply_notch_filter(filtered_data, 500, 5, Q = 100)
        
    return filtered_data
