import numpy as np
from scipy.ndimage import maximum_filter1d

def get_local_maximum_cwt(w_coefs, min_win_size=5, amp_thresh=0, 
                          is_amp_thresh_relative=False, exclude_0_scale_amp_thresh=False):
    """
    Find local maxima in wavelet coefficients.
    
    Parameters:
    w_coefs (np.ndarray): 2D array of wavelet coefficients with rows as observations and columns as scales.
    min_win_size (int): Minimum window size for detecting local maxima.
    amp_thresh (float): Amplitude threshold. Peaks below this value are ignored.
    is_amp_thresh_relative (bool): If True, threshold is relative to maximum amplitude.
    exclude_0_scale_amp_thresh (bool): If True and `is_amp_thresh_relative` is True, exclude scale 0 in thresholding.
    
    Returns:
    np.ndarray: A matrix of the same shape as `w_coefs` containing the local maxima. Non-maxima values are 0.
    """
    _, n_cols = w_coefs.shape
    scales = np.arange(n_cols)  # Assuming scales correspond to columns
    local_max = np.full_like(w_coefs, fill_value=0, dtype=int)

    # Adjust threshold if it's relative
    if is_amp_thresh_relative:
        if exclude_0_scale_amp_thresh and 0 in scales:
            mask = scales != 0
            amp_thresh = np.max(w_coefs[:, mask]) * amp_thresh
        else:
            amp_thresh = np.max(w_coefs) * amp_thresh

    # Loop over each scale (column)
    for i, scale in enumerate(scales):
        win_size = max(min_win_size, int(scale * 2 + 1))  # Ensure window size meets minimum
        filtered = maximum_filter1d(w_coefs[:, i], size=win_size, mode='constant', cval=0)
        local_max[:, i] = np.where((w_coefs[:, i] == filtered) & (w_coefs[:, i] >= amp_thresh), 1, 0)
    
    return local_max



def get_ridge(local_max, i_init=None, step=-1, i_final=0, min_win_size=5, 
              gap_th=3, skip=None, scale_to_win_size="doubleodd"):
    """
    Trace ridge lines through a local maxima matrix.

    Parameters:
    local_max (np.ndarray): 2D matrix of local maxima.
    i_init (int): Initial scale index to start tracing. Default is the last column.
    step (int): Step size for moving across scales. Default is -1 (backward).
    i_final (int): Final scale index to stop tracing. Default is the first column.
    min_win_size (int): Minimum window size for detecting peaks.
    gap_th (int): Maximum allowable gap in consecutive ridge points.
    skip (int): Column index to skip during processing. Default is None.
    scale_to_win_size (str or callable): Rule to compute window size. Options:
        - "doubleodd": `scale * 2 + 1` (default).
        - "halve": `scale / 2`.
        - Custom function `f(scale)`.

    Returns:
    dict: Ridge paths identified, where each key is a ridge and each value is a list of indices.
    """
    n_rows, n_cols = local_max.shape
    scales = np.arange(1, n_cols + 1)  # Assuming scales are integers starting from 1
    if i_init is None:
        i_init = n_cols - 1  # Default to last column if not provided

    max_ind_curr = np.where(local_max[:, i_init] > 0)[0]
    ridge_list = {idx: [idx] for idx in max_ind_curr}
    peak_status = {idx: 0 for idx in max_ind_curr}

    if skip is None:
        skip = i_init + step

    col_indices = range(i_init + step, i_final - 1, step)
    orphan_ridge_list = {}

    for col in col_indices:
        scale_j = scales[col]
        
        # Handle window size based on the scale
        if scale_to_win_size == "doubleodd":
            win_size_j = scale_j * 2 + 1
        elif scale_to_win_size == "halve":
            win_size_j = max(int(scale_j / 2), 1)
        elif callable(scale_to_win_size):
            win_size_j = scale_to_win_size(scale_j)
        else:
            raise ValueError('Invalid scaleToWinSize. Use "doubleodd", "halve", or a function(scale).')

        win_size_j = max(win_size_j, min_win_size)
        
        sel_peak_j = []
        remove_j = []
        
        for idx in max_ind_curr:
            start_idx = max(0, idx - win_size_j)
            end_idx = min(n_rows, idx + win_size_j + 1)
            candidate_indices = np.where(local_max[start_idx:end_idx, col] > 0)[0] + start_idx
            
            if len(candidate_indices) == 0:
                status = peak_status[idx]
                if status > gap_th and scale_j >= 2:
                    orphan_ridge_list[f"{col}_{idx}"] = ridge_list[idx][:-(status + 1)]
                    remove_j.append(idx)
                    continue
                else:
                    peak_status[idx] += 1
            else:
                peak_status[idx] = 0
                if len(candidate_indices) > 1:
                    closest_idx = candidate_indices[np.argmin(np.abs(candidate_indices - idx))]
                    candidate_indices = [closest_idx]

            ridge_list[idx].extend(candidate_indices)
            sel_peak_j.extend(candidate_indices)
        
        # Remove disconnected ridges
        for rid_idx in remove_j:
            del ridge_list[rid_idx]
            del peak_status[rid_idx]

        # Handle duplicate peaks
        duplicates = {x for x in sel_peak_j if sel_peak_j.count(x) > 1}
        for dup in duplicates:
            dup_indices = [key for key, val in ridge_list.items() if val[-1] == dup]
            if len(dup_indices) > 1:
                longest_path = max(dup_indices, key=lambda key: len(ridge_list[key]))
                for rid in dup_indices:
                    if rid != longest_path:
                        orphan_ridge_list[f"{col}_{rid}"] = ridge_list.pop(rid)

        # Update current peaks
        max_ind_curr = list(set(sel_peak_j))

    # Attach ridge level as part of the ridge name
    for key in ridge_list:
        ridge_list[f"1_{key}"] = ridge_list.pop(key)

    # Include orphan ridges
    ridge_list.update(orphan_ridge_list)

    # Reverse the ridge list to trace from low to high scales
    for key in ridge_list:
        ridge_list[key].reverse()

    return ridge_list




def identify_major_peaks(ms, ridge_list, w_coefs, scales=None, SNR_Th=3, peak_scale_range=5, ridge_length=32,
                         nearby_peak=False, nearby_win_size=None, win_size_noise=500, SNR_method="quantile",
                         min_noise_level=0.001, exclude_boundaries_size=None):
    if scales is None:
        scales = np.arange(1, w_coefs.shape[1] + 1)
    elif isinstance(scales, list) or isinstance(scales, np.ndarray):
        scales = np.array(scales).astype(float)
        
    if nearby_win_size is None:
        nearby_win_size = 150 if nearby_peak else 100
    if exclude_boundaries_size is None:
        exclude_boundaries_size = nearby_win_size / 2

    if ridge_length > np.max(scales):
        ridge_length = int(np.max(scales))

    if isinstance(peak_scale_range, int) or len(np.array(peak_scale_range).shape) == 0:
        peak_scale_range = scales[scales >= peak_scale_range]
    else:
        peak_scale_range = scales[(scales >= peak_scale_range[0]) & (scales <= peak_scale_range[1])]

    if min_noise_level >= 1:
        min_noise_level = {"fixed": min_noise_level}
    elif min_noise_level is None:
        min_noise_level = 0
    else:
        min_noise_level *= np.max(w_coefs)

    ridge_len = [len(r) for r in ridge_list]
    ridge_info = np.array([list(map(float, r.split("_"))) for r in ridge_list.keys()]).T
    ridge_level, mz_ind = ridge_info
    ord_indices = np.argsort(mz_ind)
    
    ridge_name = list(ridge_list.keys())[ord_indices]
    ridge_len = np.array(ridge_len)[ord_indices]
    ridge_level = ridge_level[ord_indices]
    ridge_list = {k: v for k, v in zip(ridge_name, np.array(list(ridge_list.values()))[ord_indices])}
    mz_ind = mz_ind[ord_indices].astype(int)

    peak_scale, peak_center_ind, peak_value = [], [], []

    for i, (ridge_i, level_i, len_i) in enumerate(zip(ridge_list.values(), ridge_level, ridge_len)):
        levels_i = np.arange(level_i, level_i + len_i)
        scales_i = scales[levels_i.astype(int) - 1]
        sel_ind_i = np.isin(scales_i, peak_scale_range)
        if not sel_ind_i.any():
            peak_scale.append(scales_i[0])
            peak_center_ind.append(ridge_i[0])
            peak_value.append(0)
            continue
        levels_i = levels_i[sel_ind_i]
        scales_i = scales_i[sel_ind_i]
        ridge_i = ridge_i[sel_ind_i]
        ind_i = (ridge_i - 1, levels_i - 1)
        ridge_values = w_coefs[ind_i]
        max_ind_i = np.argmax(ridge_values)
        peak_scale.append(scales_i[max_ind_i])
        peak_center_ind.append(ridge_i[max_ind_i])
        peak_value.append(ridge_values[max_ind_i])

    peak_scale = np.array(peak_scale)
    peak_center_ind = np.array(peak_center_ind).astype(int)
    peak_value = np.array(peak_value)
    
    noise = np.abs(w_coefs[:, 0])
    peak_SNR = []

    for k, ind_k in enumerate(mz_ind):
        start_k = max(ind_k - win_size_noise, 0)
        end_k = min(ind_k + win_size_noise, len(ms))
        ms_int = ms[start_k:end_k]
        
        if SNR_method == "quantile":
            noise_level_k = np.quantile(noise[start_k:end_k], 0.95)
        elif SNR_method == "sd":
            noise_level_k = np.std(noise[start_k:end_k])
        elif SNR_method == "mad":
            noise_level_k = np.median(np.abs(noise[start_k:end_k] - np.median(noise[start_k:end_k])))
        elif SNR_method == "data.mean":
            noise_level_k = np.mean(ms_int)
        elif SNR_method == "data.mean.quant":
            noise_level_k = np.mean(ms_int[ms_int < np.quantile(ms_int, 0.95)])
        else:
            raise ValueError("Invalid SNR.method")
        
        if noise_level_k < min_noise_level:
            noise_level_k = min_noise_level
        peak_SNR.append(peak_value[k] / noise_level_k)
    peak_SNR = np.array(peak_SNR)

    sel_ind1 = scales[np.floor(ridge_level + ridge_len - 1).astype(int) - 1] >= ridge_length

    if nearby_peak:
        sel_ind1 = np.where(sel_ind1)[0]
        temp_ind = np.hstack([np.where((mz_ind >= mz_ind[i] - nearby_win_size) & 
                                       (mz_ind <= mz_ind[i] + nearby_win_size))[0] for i in sel_ind1])
        sel_ind1 = np.isin(range(len(mz_ind)), temp_ind)

    sel_ind2 = peak_SNR > SNR_Th
    sel_ind3 = ~((mz_ind <= exclude_boundaries_size) | 
                 (mz_ind >= len(w_coefs) - exclude_boundaries_size))
    sel_ind = sel_ind1 & sel_ind2 & sel_ind3

    return {
        "peakIndex": mz_ind[sel_ind],
        "peakValue": peak_value[sel_ind],
        "peakCenterIndex": peak_center_ind[sel_ind],
        "peakSNR": peak_SNR[sel_ind],
        "peakScale": peak_scale[sel_ind],
        "potentialPeakIndex": mz_ind[sel_ind1 & sel_ind3],
        "allPeakIndex": mz_ind,
    }

