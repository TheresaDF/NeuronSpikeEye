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

CAP_length = lambda x: len(x) if type(x) == list else 0

def count_caps(simulator : SimulateData, filtered_signal : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Function that estimates the number of CAPs in the signal """

    # allocate memory for the counts
    all_est_counts = np.zeros((simulator.num_channels, int(simulator.duration * simulator.stim_freq)))
    all_true_counts = np.zeros((simulator.num_channels, int(simulator.duration * simulator.stim_freq)))

    # loop over all channels
    for channel in tqdm(range(simulator.num_channels)):
        # find the SA and bin accordingly
        peaks, _ = find_peaks(simulator.signal[:, channel], height = 30, distance = 300000 / (simulator.stim_freq * simulator.duration))
        bins = bin_data(filtered_signal[:, channel], peaks).T 

        # loop over all bins
        for bin_idx in range(bins.shape[1]):
            # apply wavelet transform
            coefficients, _ = pywt.cwt(bins[:, bin_idx], scales=np.arange(1, 128), wavelet='cgau2', sampling_period=1/30000)
            
            # find local maxima at all scales 
            scales = np.arange(1, 128)
            snakes = np.zeros((len(scales), bins.shape[0]))

            for coef in range(len(scales)): 
                rms = np.sqrt(np.mean(np.abs(coefficients[coef])**2))
                peaks_coef, _ = find_peaks(np.abs(coefficients[coef]), height = 1.5*rms, distance = 30)
                
                snakes[coef, peaks_coef] = 1

            # count the number of peaks in the bin (length and outlier)
            labels = label(snakes)
            components = np.bincount(labels.flat)[1:]
            rms_components = np.sqrt(np.mean(components**2))

            # save the number of estimates bins
            all_est_counts[channel, bin_idx] = len(np.where((components > 4 * rms_components) & (components > 20))[0])
            
            # get the number of true counts 
            all_true_counts[channel, bin_idx] = CAP_length(simulator.CAP_indices[bin_idx][channel])

    return all_est_counts, all_true_counts
