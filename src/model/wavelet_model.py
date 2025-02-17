from src.data.preprocess_utils import bin_data
from skimage.morphology import binary_erosion
from scipy.signal import find_peaks
from src.model.robust_ntf import robust_ntf
from tensorly.tenalg import outer
from tqdm import tqdm
import numpy as np 
import torch 
import pywt 


def clean_scalograms(scalograms : np.ndarray) -> np.ndarray:
    # convert to torch tensor
    scalograms_torch = torch.tensor(scalograms).cuda()

    # define parameters 
    rank = 91 
    beta = 1
    reg_val = 30
    tol = 1e-3

    # run robust NTF
    factors, _, _ = robust_ntf(scalograms_torch, rank=rank, beta=beta, reg_val=reg_val, tol=tol)

    # reconstruct 
    rntf_recon = torch.zeros(scalograms.shape)
    for i in range(rank):
        rntf_recon = rntf_recon + outer([factors[0][:,i],
                                        factors[1][:,i],
                                        factors[2][:,i]])

    rntf_recon = rntf_recon.numpy()

    return rntf_recon


def get_accepted_coefficients(coefficients : np.ndarray, scales : np.ndarray, ratio : float) -> tuple[np.ndarray, np.ndarray]:
    accepted_coefficients = np.zeros_like(coefficients)

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
            log_gamma = 36 * ratio + np.log(p_noise / p_spikes)

            # compute acceptance threshold
            theta = mu / 2 + sigma_j**2 / mu * log_gamma

            # apply acceptance threshold
            accepted_coefficients[j] = x * (np.abs(x) > theta)

    return accepted_coefficients

def get_spike_indicators(accepted_coefficients : np.ndarray) -> np.ndarray:
    # get locations of spikes 
    col_sum = np.sum(np.abs(accepted_coefficients), axis = 0)
    mask = col_sum > np.mean(accepted_coefficients[accepted_coefficients > 0]) * accepted_coefficients.shape[0] / 3 
    col_sum[~mask] = 0 
    spike_indicators = col_sum.astype(bool).astype(int)

    return spike_indicators

def parse(spike_indicators : np.ndarray, fs : int, width : tuple): 
    # define refractory period (also defined in Jesper's paper)
    refract_len = 30 

    # merge spikes closer than merge 
    merge = np.mean(width) * fs   
    
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

    

def count_caps_wavelet(orig_signal : np.ndarray, filtered_signal : np.ndarray, duration : int = 10, stim_freq : int = 10, bin : bool = True) -> np.ndarray:
    """ Function that estimates the number of CAPs in the signal """
    # get number of channels 
    num_channels = filtered_signal.shape[1]

    # allocate memory for the counts
    all_est_counts = np.zeros((num_channels, int(duration * stim_freq))) if bin else np.zeros((num_channels, 1))

    # loop over all channels
    for channel in tqdm(range(num_channels)):
        scalograms = np.zeros((int(stim_freq * duration), 127, 2400))
        if bin: 
            # find the SA and bin accordingly
            peaks, _ = find_peaks(orig_signal[:, channel], height = 300, distance = 300000 / (stim_freq * duration) - stim_freq * duration)
            bins = bin_data(filtered_signal[:, channel], peaks).T 

            # loop over all bins
            for bin_idx in range(bins.shape[1]):
                # apply wavelet transform
                coefficients, _ = pywt.cwt(bins[:, bin_idx], scales=np.arange(1, 128), wavelet='cgau1', sampling_period=1/30000)

                # save the scalograms
                scalograms[bin_idx] = np.abs(coefficients)

        # clean the scalograms
        scalograms = clean_scalograms(scalograms)
        for i in range(scalograms.shape[0]):
            # get accepted coefficients
            accepted_coefficients = get_accepted_coefficients(scalograms[i], scales=np.arange(1, 128), ratio = 0.1)

            # get spike indicators 
            spike_indicators = get_spike_indicators(accepted_coefficients)

            # merge and parse the spikes
            TE = parse(spike_indicators, fs=30, width=(3, 9))

            # save the number of estimates caps
            all_est_counts[channel, i] = len(TE)

        # if spontaneous data 
        else: 
            # apply wavelet transform
            coefficients, _ = pywt.cwt(filtered_signal[:, channel], scales=np.arange(1, 128), wavelet='cgau1', sampling_period=1/30000)
            
            # get accepted coefficients
            spike_indicators, _ = get_accepted_coefficients(coefficients, scales=np.arange(1, 128), ratio = 0.1)

            # merge and parse the spikes
            TE = parse(spike_indicators, fs=30, width=(3, 9))

            # save the number of estimates caps
            all_est_counts[channel] = len(TE)


    return all_est_counts
