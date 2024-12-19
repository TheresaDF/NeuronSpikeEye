import numpy as np
from scipy.ndimage import maximum_filter1d

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
    orphan_ridge_name = []

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
