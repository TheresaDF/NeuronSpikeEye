from src.data.create_simulated_data import SimulateData
from scipy.signal import find_peaks
from sklearn.svm import SVR 
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


CAP_length = lambda x: len(x) if type(x) == list else 0

def make_matrices(simulator : SimulateData, filtered_signal : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Make matrices for training and testing
    """

    y = np.array(([np.sum([CAP_length(simulator.CAP_indices[i][channel]) for i in range(int(simulator.stim_freq * simulator.duration))]) 
                   for channel in range(simulator.num_channels)])).ravel()

    X = np.zeros((simulator.num_channels, simulator.stim_freq * simulator.duration*2400))
    for channel in range(simulator.num_channels):
        peaks, _ = find_peaks(filtered_signal[:, channel], height = 300, distance = 300000 / (simulator.stim_freq * simulator.duration) - simulator.stim_freq * simulator.duration)
        data = bin_data(filtered_signal[:, channel], peaks)
        X[channel] = data.ravel()

    # normalize X
    X = (X - np.mean(X, axis = 1).reshape(-1, 1)) / np.std(X, axis = 1).reshape(-1, 1)
    return X, y 

def count_caps_svm(simulator_train : SimulateData, simulator_test : SimulateData, filtered_signal_train : np.ndarray, filtered_signal_test : np.ndarray) -> np.ndarray:
    # construct matrices
    X_train, y_train = make_matrices(simulator_train, filtered_signal_train)
    X_test, _ = make_matrices(simulator_test, filtered_signal_test)

    # initialize the regressor 
    regressor = SVR(kernel = "linear")

    # train the regressor
    regressor.fit(X_train, y_train)

    # predict the test data
    y_pred_test = regressor.predict(X_test)

    return y_pred_test