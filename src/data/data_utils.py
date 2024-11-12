from scipy.signal import find_peaks
import numpy as np
from scipy.stats import lognorm
from scipy.fft import fft, fftfreq, ifft 

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


def get_pli_indices(bins : np.ndarray) -> tuple[float, list, list]:
    all_peaks = []
    idx_p = []
    idx_m = []

    for i in range(bins.shape[0]):
        # extract segment 
        segment = bins[i]

        # find both negative and positive peaks 
        peaks_p, _ = find_peaks(segment, distance = 20*30 - 10) # 20ms
        peaks_m, _ = find_peaks(-segment, distance = 20*30 - 10) # 20ms

        # save indices 
        idx_p.append(peaks_p)
        idx_m.append(peaks_m)

        # evaluate peaks in segment 
        amp_p = segment[peaks_p]
        amp_m = np.abs(segment[peaks_m])
        
        # save result 
        all_peaks.append(amp_p)
        all_peaks.append(amp_m)

    # unravel 
    all_peaks = [item for sublist in all_peaks for item in sublist]
    amp = np.mean(all_peaks)

    return amp, idx_p, idx_m

def make_pli_spike(amp : float) -> np.ndarray:
    """ Create spike based on lognormal distribution """
    spike = np.flip(lognorm.pdf(np.linspace(0, 4, 100), 0.5, 0, 0.45))
    spike = spike / np.max(spike) * amp 

    return spike 

def insert_single_spike(spike, loc, n, sign = 1):
    """
    Create segment of spikes 
    """

    segment = np.zeros(n)
    spike_max = np.argmax(spike) - 2

    if loc - spike_max < 0:
        d = spike_max - loc 
        segment[0:loc - spike_max + len(spike)] += spike[d:]
    elif loc - spike_max + len(spike) > len(segment):
        segment[loc - spike_max:loc - spike_max + len(spike)] += spike 
    else:
        segment[loc - spike_max:loc - spike_max + len(spike)] += spike 

    segment *= sign 
    return segment 

def create_spike_segment(n : int, bin : np.ndarray, spike : np.ndarray, idx_p : list, idx_m : list) -> np.ndarray:
    spike_segment = np.zeros(n)
    for i in range(len(idx_p)):
        spike_segment += insert_single_spike(spike, idx_p[i], len(bin))
        spike_segment += insert_single_spike(spike, idx_m[i], len(bin), sign = -1)

    return spike_segment 

def bin_data(data : np.ndarray) -> np.ndarray:
    pass 

def remove_spike_segment(data : np.ndarray) -> np.ndarray:
    """ Remove spike segment from bin """
    
    bins = bin_data(data)
    bins = high_pass(bins)
    amp, idx_p, idx_m = get_pli_indices(bins)
    spike = make_pli_spike(amp)
    spike_segment = create_spike_segment(data.shape[1], bins, spike, idx_p, idx_m)
    
    return bin - spike_segment
