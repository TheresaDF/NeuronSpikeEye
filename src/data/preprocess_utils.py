import numpy as np 
from scipy.signal import wiener, find_peaks 
from matplotlib import pyplot as plt 
from scipy.fft import fft, fftfreq, ifft 
from sklearn.decomposition import FastICA  
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
    ica = FastICA(n_components = n_components, whiten_solver = "svd", max_iter = 1000, random_state = 42)
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

def bin_data(channel, peaks):
    """
    Bin data into 80ms bins
    """
    binned_data = np.zeros((100, 2400))
    for c, peak in enumerate(peaks):
        if c == 100: break 
        if (c == 99) & (peak+2700 > len(channel)):
            binned_data[c, :len(channel) - peak] = channel[peak:]
        else: 
            binned_data[c] = channel[peak+300:peak+2700]
    
    return binned_data

def find_bad_channels(data : np.ndarray, peaks : np.ndarray, n_comp : int) -> tuple[int, float]:
    """ Function to find bad channels using ICA"""

    acfs_all = np.zeros((n_comp, 3))
    for i in range(n_comp):
        # bin segment 
        bins = bin_data(data[:, i], peaks).ravel()

        acf_tmp = acf(bins, nlags=20*30*3)
        acfs_all[i, 0] = acf_tmp[20*30]
        acfs_all[i, 1] = acf_tmp[20*30 * 2]
        acfs_all[i, 2] = acf_tmp[20*30 * 3] 
    
    # bad_components = np.where(np.mean(acfs_all, axis = 1) > np.percentile(np.mean(acfs_all, axis = 1), 75))[0]
    bad_component = np.argmax(np.mean(acfs_all, axis = 1))
    value = np.max(np.mean(acfs_all, axis = 1))

    return bad_component, value 


def butter_lowpass(data : np.ndarray, cutoff : int = 5000, fs : int = 30000, order=20):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def smooth_signal(data : np.ndarray, window_len : int = 15) -> np.ndarray:
    """ Function to smooth the signal """
    for i in range(data.shape[1]):
        data[:, i] = wiener(data[:, i], window_len)

    return data 

def compute_num_components(data, threshold = 0.95, to_plot = False) -> np.ndarray:
    """
    Function to compute the number of components to keep
    """
   
    n_comp = np.arange(1, data.shape[1]+1)
    loss = np.zeros(len(n_comp))
    for i, n in enumerate(n_comp):
        ica, ica_components = perform_ICA(data, n_components = n)

        # reconstruct 
        reconstruction = ica.inverse_transform(ica_components)

        # compute scaling factor for scaled MSE 
        alpha = np.mean(data * reconstruction) / np.mean(reconstruction ** 2)

        # compute reconstruction loss 
        loss_current = np.mean((data - alpha * reconstruction) ** 2)
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

def get_acf_signal(signal : np.ndarray, stim_freq : int = 10, length : int = 300000, duration : int = 10) -> float:
    """
    Function to compute the acf of the signal
    """
    all_acfs = np.zeros(32)
    for channel in range(32): 
        peaks, _ = find_peaks(signal[:, channel], height = 300, distance = length / (stim_freq * duration))
        signal_bins = bin_data(signal[:, channel], peaks).ravel()
        acf_signal_tmp = acf(signal_bins, nlags=20*30*3)
        acf_signal = np.mean([acf_signal_tmp[20*30], acf_signal_tmp[20*30 * 2], acf_signal_tmp[20*30 * 3]])
        all_acfs[channel] = acf_signal

    return np.mean(all_acfs)

def filter(data : np.ndarray, stim_freq : int = 10, length : int = 300000, duration : int = 10) -> np.ndarray:
    """ Use ICA to filtef data"""

    # smooth and filter the data
    data = remove_drift(data)
    data = butter_lowpass(data)
    data = smooth_signal(data, window_len=5)

    # find acf in original signal 
    data_filtered = data 
    peaks, _ = find_peaks(data[:, 0], height = 300, distance = length / (stim_freq * duration))
    acf_signal = get_acf_signal(data_filtered, stim_freq = stim_freq, length = length, duration = duration)

    print(f"Original ACF : {acf_signal}")
    
    # define parameters 
    n_comp = 32 
    ratio = 1 
    count = 0 
    # start loop 
    while (ratio > 0.65) & (count < 10):
        # perform ICA
        ica, ica_components = perform_ICA(data_filtered, n_comp)

        # find worst 59 hz component 
        bad_component, acf_curr = find_bad_channels(ica_components, peaks, n_comp=n_comp)

        # remove component 
        data_filtered = remove_ica_components(ica, ica_components, bad_component)
        acf_curr = get_acf_signal(data_filtered)

        # decrement number of components 
        n_comp -= 1 

        # compute difference in acf 
        ratio = acf_curr / acf_signal 
        count += 1         

        print("new acf : ", acf_curr)
        print(f"ratio : {ratio}")

    # smooth end result again 
    return data_filtered