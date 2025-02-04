from src.data.preprocess_utils import bin_data
from skimage.morphology import binary_erosion
from scipy.signal import find_peaks
from tqdm import tqdm
import numpy as np 
import pywt 


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
            log_gamma = 36 * 0.1 + np.log(p_noise / p_spikes)

            # compute acceptance threshold
            theta = mu / 2 + sigma_j**2 / mu * log_gamma

            # apply acceptance threshold
            accepted_coefficients[j] = x * (np.abs(x) > theta)

        # needed for arrival time analysis (not used yet)
        non_zero_indices = accepted_coefficients[j] != 0
        non_zero_indices = spike_indicators | non_zero_indices
        spike_indicators = non_zero_indices

    # clean up spike indicators 
    spikes_eroded = binary_erosion(spike_indicators.astype(int), np.ones(5))

    return spikes_eroded.astype(int), accepted_coefficients

def parse(spike_indicators : np.ndarray, fs : int, width : tuple): 
    # define refractory period (also defined in Jesper's paper)
    refract_len = 30 

    # merge spikes closer than merge 
    merge = round(np.mean(width) * fs)       
    
    # discard spikes at beginning and end
    spike_indicators[0] = 0 
    spike_indicators[-1] = 0 

    # locate the ones 
    ind_ones = np.where(spike_indicators == 1)[0]

    if len(ind_ones) == 0:
        TE=[]
    else: 
        tmp=np.diff(spike_indicators);  
        n_sp = np.sum(tmp == 1); 
        
        # index of the beginning of a spike
        lead_t = np.where(tmp == 1)[0]

        # index of the end of the spike
        lag_t = np.where(tmp == -1)[0] 

        te = np.zeros(n_sp)
        for i in range(n_sp):
            te[i] = np.ceil(np.mean([lead_t[i], lag_t[i]]))
        
        # init counter 
        i = 0 
        while True: 
            if i > len(te) - 2: 
                break
            else: 
                diff = te[i+1] - te[i]
                if (diff < refract_len) & (diff > merge):  
                    # discard the spike
                    te = np.delete(te, i+1)
                elif diff <= merge: 
                    # merge the spikes
                    te[i] = np.ceil(np.mean([te[i], te[i+1]]))
                    
                    # discard the spike
                    te = np.delete(te, i+1)
                else: 
                    i += 1 
        TE = te 

    return TE

    

def count_caps_wavelet(orig_signal : np.ndarray, filtered_signal : np.ndarray, duration : int = 10, stim_freq : int = 10) -> np.ndarray:
    """ Function that estimates the number of CAPs in the signal """
    # get number of channels 
    num_channels = filtered_signal.shape[1]

    # allocate memory for the counts
    all_est_counts = np.zeros((num_channels, int(duration * stim_freq)))

    # loop over all channels
    for channel in range(num_channels):
        # find the SA and bin accordingly
        peaks, _ = find_peaks(orig_signal[:, channel], height = 300, distance = 300000 / (stim_freq * duration) - stim_freq * duration)
        bins = bin_data(filtered_signal[:, channel], peaks).T 

        # loop over all bins
        for bin_idx in range(bins.shape[1]):
            # apply wavelet transform
            coefficients, _ = pywt.cwt(bins[:, bin_idx], scales=np.arange(1, 128), wavelet='cgau1', sampling_period=1/30000)
            
            # get accepted coefficients
            spike_indicators, _ = get_accepted_coefficients(coefficients, scales=np.arange(1, 128))

            # merge and parse the spikes
            TE = parse(spike_indicators, fs=30, width=(3, 9)) # from how the simulated data is constructed 
            
            # save the number of estimates caps 
            all_est_counts[channel, bin_idx] = len(TE)


    return all_est_counts
