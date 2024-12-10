import pywt 
import numpy as np 
from tqdm import tqdm
from skimage.measure import label   
from scipy.signal import find_peaks
from dataset_utils import read_ns5_file
from create_simulated_data import SimulateData


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

def count_caps(configurations) -> tuple[np.ndarray, np.ndarray]:
    """ Function that estimates the number of CAPs in the signal """

    # allocate memory for the counts
    all_est_counts = np.zeros((configurations.num_channels, configurations.duration * configurations.stim_freq))
    all_true_counts = np.zeros((configurations.num_channels, configurations.duration * configurations.stim_freq))

    # loop over all channels
    for channel in range(configurations.num_channels):
        # find the SA and bin accordingly
        peaks, _ = find_peaks(configurations.signal[:, channel], height = 30, distance = 300000 / (configurations.stim_freq * configurations.duration))
        bins = bin_data(configurations.filtered_signal[:, channel], peaks).T 

        # allocate memory for the counts
        counts = np.zeros(configurations.duration * configurations.stim_freq)

        # loop over all bins
        for bin_idx in tqdm(range(len(counts))):
            # apply wavelet transform
            coefficients, _ = pywt.cwt(bins[:, bin_idx], scales=np.arange(1, 128), wavelet='cgau1', sampling_period=1/30000)
            
            # threshold the coefficients 
            coefs_real = np.real(np.abs(coefficients))
            threshold = np.mean(coefs_real) + 4 * np.std(coefs_real)
            significant_coefficients = np.real(np.where(np.abs(coefficients) > threshold, coefficients, 0))
            label_image = label(np.abs(significant_coefficients) > 0)

            # count the number of peaks in the bin
            counts[bin_idx] = np.max(label_image)

        # get true number of peaks
        true_counts = [CAP_length(f) for f in configurations.CAP_indices[:, channel]]

        # save the counts 
        all_est_counts[channel] = counts
        all_true_counts[channel] = true_counts

    return all_est_counts, all_true_counts

def compare_cap_count(configurations) -> None:    

    # compute estimated cap count 
    est_counts, true_counts = count_caps(configurations, CAP_length)



if __name__ == "__main__":
    # load configuration 
    configurations = np.load("configurations.npy", allow_pickle=True).item()

    compare_cap_count(configurations)