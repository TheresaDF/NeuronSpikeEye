import pywt 
import numpy as np 
from tqdm import tqdm
from skimage.measure import label   
from scipy.signal import find_peaks
from src.data.create_simulated_data import SimulateData


def bin_data(channel : np.ndarray, peaks : list) -> np.ndarray:
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


def get_accepted_coefficients(coefficients : np.ndarray, scales : np.ndarray) -> np.ndarray:
    accepted_coefficients = np.zeros_like(coefficients)
    spike_indicators = np.zeros(coefficients.shape[1], dtype = bool)

    # loop over scales 
    for j in range(len(scales)): 
        # extract current coefficients
        x = coefficients[j]

        # estimate noise level 
        sigma_j = np.median(np.abs(x - np.mean(x))) / 0.6745 

        # compute hard threshold 
        T = np.sqrt(2 * np.log(len(x))) * sigma_j

        # find indices of coefficients that exceed threshold
        index = np.where(np.abs(x) > T)[0]

        if len(index) > 0: 
            # compute sample mean 
            mu = np.mean(np.abs(x[index]))

            # compute probabilities
            p_spikes = len(index) / len(x)
            p_noise = 1 - p_spikes

            # compute gamma
            log_gamma = 36.7368 * (0) + np.log(p_noise / p_spikes)

            # compute acceptance threshold
            theta = mu / 2 + sigma_j**2 / mu * log_gamma

            # apply acceptance threshold
            accepted_coefficients[j] = x * (np.abs(x) > theta)

        # needed for arrival time analysis (not used yet)
        non_zero_indices = accepted_coefficients[j] != 0
        non_zero_indices = spike_indicators | non_zero_indices
        spike_indicators = non_zero_indices

    return accepted_coefficients
    

def count_caps_wavelet(simulator : SimulateData, filtered_signal : np.ndarray) -> np.ndarray:
    """ Function that estimates the number of CAPs in the signal """

    # allocate memory for the counts
    all_est_counts = np.zeros((simulator.num_channels, int(simulator.duration * simulator.stim_freq)))

    # loop over all channels
    for channel in tqdm(range(simulator.num_channels)):
        # find the SA and bin accordingly
        peaks, _ = find_peaks(simulator.signal[:, channel], height = 300, distance = 300000 / (simulator.stim_freq * simulator.duration) - simulator.stim_freq * simulator.duration)
        bins = bin_data(filtered_signal[:, channel], peaks).T 

        # loop over all bins
        for bin_idx in range(bins.shape[1]):
            # apply wavelet transform
            coefficients, _ = pywt.cwt(bins[:, bin_idx], scales=np.arange(1, 128), wavelet='cgau2', sampling_period=1/30000)
            
            # get accepted coefficients
            accepted_coefficients = get_accepted_coefficients(coefficients, scales=np.arange(1, 128))

            # find number of connected components
            labels = label(np.abs(accepted_coefficients))
            
            # save the number of estimates caps 
            all_est_counts[channel, bin_idx] = np.max(labels)


    return all_est_counts
