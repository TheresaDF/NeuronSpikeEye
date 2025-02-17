from src.data.preprocess_utils import bin_data 
from scipy.signal import find_peaks
from tqdm import tqdm
import numpy as np



def count_caps_baseline(orig_signal : np.ndarray, filtered_signal : np.ndarray, duration : int = 10, stim_freq : int = 10, bin : bool = True) -> tuple[np.ndarray]:
    """ Function that estimates the number of CAPs in the signal """
    
    # get number of channels 
    num_channels = filtered_signal.shape[1]

    # allocate memory for the counts
    all_est_counts = np.zeros((num_channels, int(duration * stim_freq))) if bin else np.zeros((num_channels, 1))

    # loop over all channels
    for channel in tqdm(range(num_channels)):
        if bin: 
            # find the SA and bin accordingly
            peaks, _ = find_peaks(orig_signal[:, channel], height = 300, distance = 300000 / (stim_freq * duration) - stim_freq * duration)
            bins = bin_data(filtered_signal[:, channel], peaks).T 

            # loop over all bins
            for bin_idx in range(bins.shape[1]):
                # find threshold of 4.5 times rms 
                rms = np.sqrt(np.mean(bins[:, bin_idx]**2))
                peaks_p, _ = find_peaks(bins[:, bin_idx], height = 4.5*rms, distance = 30)
                peaks_m, _ = find_peaks(-bins[:, bin_idx], height = 4.5*rms, distance = 30)

                # save results 
                all_est_counts[channel, bin_idx] = len(peaks_p) + len(peaks_m)

        # if spontaneous channel 
        else: 
            # find the threshold of 4.5 times rms 
            rms = np.sqrt(np.mean(filtered_signal[:, channel]**2))
            peaks_p, _ = find_peaks(filtered_signal[:, channel], height = 4.5*rms, distance = 30)
            peaks_m, _ = find_peaks(-filtered_signal[:, channel], height = 4.5*rms, distance = 30)

            # save results 
            all_est_counts[channel, 0] = len(peaks_p) + len(peaks_m)


    return all_est_counts