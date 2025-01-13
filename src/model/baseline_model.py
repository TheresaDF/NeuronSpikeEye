from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter
from scipy.signal import find_peaks
from tqdm import tqdm
import numpy as np

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


def count_caps_baseline(orig_signal : np.ndarray, filtered_signal : np.ndarray, num_channels : int = 32, duration : int = 10, stim_freq : int = 10) -> tuple[np.ndarray]:
    """ Function that estimates the number of CAPs in the signal """

    # allocate memory for the counts
    all_est_counts = np.zeros((num_channels, int(duration * stim_freq)))

    # loop over all channels
    for channel in tqdm(range(num_channels)):
        # find the SA and bin accordingly
        peaks, _ = find_peaks(orig_signal[:, channel], height = 30, distance = 300000 / (stim_freq * duration))
        bins = bin_data(filtered_signal[:, channel], peaks).T 

        # loop over all bins
        for bin_idx in range(bins.shape[1]):
            # find threshold of 4.5 times rms 
            rms = np.sqrt(np.mean(bins[:, bin_idx]**2))
            peaks_p, _ = find_peaks(bins[:, bin_idx], height = 4.5*rms, distance = 30)
            peaks_m, _ = find_peaks(-bins[:, bin_idx], height = 4.5*rms, distance = 30)

            # save results 
            all_est_counts[channel, bin_idx] = len(peaks_p) + len(peaks_m)

    return all_est_counts