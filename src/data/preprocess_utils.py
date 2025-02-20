import neo 
import numpy as np 
from scipy.signal import wiener, find_peaks 
from matplotlib import pyplot as plt 
from scipy.fft import fft, fftfreq, ifft 
from sklearn.decomposition import FastICA  
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import acf


def read_ns5_file(filename : str) -> tuple[np.ndarray, np.ndarray]:
    """ Function to read files of .ns5 format and return the data and time values."""
    reader = neo.io.BlackrockIO(filename = filename, verbose = True)
    times = reader.read_block(0).segments[0].analogsignals[0].times 
    data = reader.read_block(0).segments[0].analogsignals[0].magnitude

    return times, data


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
    binned_data = np.zeros((len(peaks), 2400))
    for c, peak in enumerate(peaks):
        if (c == len(peaks)-1) & (peak+2700 > len(channel)):
            binned_data[c, :len(channel) - peak] = channel[peak:]
        else: 
            binned_data[c] = channel[peak+300:peak+2700]
    
    return binned_data


def find_bad_channels(data : np.ndarray, peaks : np.ndarray, n_comp : int, bin : bool = True) -> tuple[int, float]:
    """ Function to find bad channels using ICA"""

    acfs_all = np.zeros((n_comp, 3))
    for i in range(n_comp):
        if bin: 
            # bin segment 
            bins = bin_data(data[:, i], peaks).ravel()
        else: 
            bins = data[:, i]

        acf_tmp = acf(bins, nlags=900)
        acfs_all[i, 0] = np.abs(acf_tmp[300])
        acfs_all[i, 1] = np.abs(acf_tmp[600])
        acfs_all[i, 2] = np.abs(acf_tmp[900] )
    
    # bad_components = np.where(np.mean(acfs_all, axis = 1) > np.percentile(np.mean(acfs_all, axis = 1), 75))[0]
    bad_component = np.argmax(np.mean(acfs_all, axis = 1))
    value = np.max(np.mean(acfs_all, axis = 1))

    return bad_component, value 


def butter_lowpass(data : np.ndarray, cutoff : int = 5000, fs : int = 30000, order : int =20):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def smooth_signal(data : np.ndarray, window_len : int = 15) -> np.ndarray:
    """ Function to smooth the signal """
    for i in range(data.shape[1]):
        data[:, i] = wiener(data[:, i], window_len)

    return data 

def get_acf_signal(signal : np.ndarray, stim_freq : int = 10, length : int = 300000, duration : int = 10, bin : bool = True) -> float:
    """
    Function to compute the acf of the signal
    """
    all_acfs = np.zeros(signal.shape[1])
    for channel in range(signal.shape[1]): 
        if bin: 
            peaks, _ = find_peaks(signal[:, channel], height = 300, distance = length / (stim_freq * duration) - stim_freq * duration)

            # some channels are broken 
            try: 
                signal_bins = bin_data(signal[:, channel], peaks).ravel()
            except: 
                all_acfs[channel] = np.nan
                continue
        else:
            signal_bins = signal[:, channel]

        acf_signal_tmp = acf(signal_bins, nlags=20*30*3)
        acf_signal = np.mean([acf_signal_tmp[20*30], acf_signal_tmp[20*30 * 2], acf_signal_tmp[20*30 * 3]])
        all_acfs[channel] = acf_signal

    return np.nanmean(all_acfs)

def filter(data : np.ndarray, stim_freq : int = 10, length : int = 300000, duration : int = 10, threshold : float = 0.65, bin : bool = True) -> tuple[np.ndarray, list]:
    """ Use ICA to filtef data"""

    # smooth and filter the data
    data = remove_drift(data)
    data = butter_lowpass(data)
    data = smooth_signal(data, window_len=5)

    # find acf in original signal 
    data_filtered = data
    idx = []
    if bin:  
        peaks = []
        for channel in range(32):
            peaks, _ = find_peaks(data_filtered[:, channel], height = 1000, distance = length / (stim_freq * duration) - stim_freq * duration)
            if len(peaks) < 99:
                idx.append(channel)
    else: 
        peaks = []
        
    # remove broken channels
    data_filtered = np.delete(data_filtered, idx, axis = 1)
    acf_signal = get_acf_signal(data_filtered, stim_freq = stim_freq, length = length, duration = duration)

    print(f"Original ACF : {acf_signal}")
    
    # define parameters 
    n_comp = data_filtered.shape[1]
    ratio = 1 
    count = 0 

    # start loop 
    while (ratio > threshold) & (count < 10):
        # perform ICA
        ica, ica_components = perform_ICA(data_filtered, n_comp)

        # find worst 59 hz component 
        bad_component, acf_curr = find_bad_channels(ica_components, peaks, n_comp=n_comp, bin = bin)

        # remove component 
        data_filtered = remove_ica_components(ica, ica_components, bad_component)
        acf_curr = get_acf_signal(data_filtered, bin = bin)

        # decrement number of components 
        n_comp -= 1 

        # compute difference in acf 
        ratio = acf_curr / acf_signal 
        count += 1         

        print("new acf : ", acf_curr)
        print(f"ratio : {ratio}")

    # smooth end result again 
    return data_filtered, idx 