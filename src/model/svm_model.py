from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import bin_data
from scipy.signal import find_peaks
from sklearn.svm import SVR 
import numpy as np
import pywt 




CAP_length = lambda x: len(x) if type(x) == list else 0

def make_matrices(simulator : SimulateData, filtered_signal : np.ndarray, duration : int = 10, stim_freq : int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Make matrices for training and testing
    """
    num_channels = filtered_signal.shape[1]
    if simulator is not None:
        y = np.array(([np.sum([CAP_length(simulator.CAP_indices[i][channel]) for i in range(int(stim_freq * duration))]) 
                    for channel in range(num_channels)])).ravel()
    else: 
        y = np.zeros(num_channels)

    X = np.zeros((num_channels, int(stim_freq * duration*2400)))
    for channel in range(num_channels):
        peaks, _ = find_peaks(filtered_signal[:, channel], height = 300, distance = 300000 / (stim_freq * duration) - stim_freq * duration)
        data = bin_data(filtered_signal[:, channel], peaks)
        X[channel] = data.ravel()

    return X, y 

def convert_to_scaleogram(signal : np.ndarray) -> np.ndarray:
    scales = np.arange(1, 128)
    X_wavelet = np.zeros((signal.shape[0], len(scales)))

    # convert to scaleogram
    for i in range(signal.shape[0]):
        coefs = pywt.cwt(signal[i], scales, "cgau1")[0]

        # average out time dimension 
        X_wavelet[i] = np.mean(np.abs(coefs), axis = 1)

    return X_wavelet

def count_caps_svm(simulator_train : SimulateData, filtered_signal_train : np.ndarray, filtered_signal_test : np.ndarray) -> np.ndarray:
    # construct matrices
    X_train, y_train = make_matrices(simulator_train, filtered_signal_train)
    X_test, _ = make_matrices(None, filtered_signal_test)

    # convert to scaleogram
    X_train = convert_to_scaleogram(X_train)
    X_test = convert_to_scaleogram(X_test)

    # normalize data 
    X_train = X_train / np.max(X_train, axis = 1)
    X_test = X_test / np.max(X_test, axis = 1)
    y_max = np.max(y_train)
    y_train = y_train / y_max 

    # initialize the regressor 
    regressor = SVR(kernel = "linear")

    # train the regressor
    regressor.fit(X_train, y_train) 

    # predict the test data
    y_pred_test = regressor.predict(X_test) 

    return y_pred_test * y_max