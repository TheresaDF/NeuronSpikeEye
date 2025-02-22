from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import bin_data
from scipy.signal import find_peaks, welch 
from sklearn.svm import SVR 
import numpy as np


CAP_length = lambda x: len(x) if type(x) == list else 0

def make_matrices(simulator : SimulateData, filtered_signal : np.ndarray, duration : int = 10, stim_freq : int = 10, bin = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Make matrices for training and testing
    """
    # get the number of caps 
    num_channels = filtered_signal.shape[1]
    if (simulator is not None) & (simulator.CAP_indices is not None):
        y = np.array(([np.sum([CAP_length(simulator.CAP_indices[i][channel]) for i in range(int(stim_freq * duration))]) 
                    for channel in range(num_channels)])).ravel()
    else: 
        y = np.zeros(num_channels)

    if bin: 
        # gather data without SA
        X = np.zeros((num_channels, int(stim_freq * duration*2400))) 
        for channel in range(num_channels):
            peaks, _ = find_peaks(filtered_signal[:, channel], height = 300, distance = 300000 / (stim_freq * duration) - stim_freq * duration)
            data = bin_data(filtered_signal[:, channel], peaks)
            X[channel, :min(len(data.ravel()), int(stim_freq * duration*2400))] = data.ravel()[:int(stim_freq * duration*2400)]
    else: 
        X = filtered_signal.T 
    return X, y 

def convert_to_frequency(signal : np.ndarray) -> np.ndarray:
    # convert to frequency comain
    for i in range(signal.shape[0]):
        f, Sxx = welch(signal[i], fs = 30000, nperseg = 256)
        if i == 0 : 
            X_freq = np.zeros((signal.shape[0], len(Sxx[f < 5000])))
        
        X_freq[i] = Sxx[f < 5000]
    return X_freq


def count_caps_svm(simulator_train : SimulateData, filtered_signal_train : np.ndarray, filtered_signal_test : np.ndarray, bin : bool = True) -> np.ndarray:
    # construct matrices
    X_train, y_train = make_matrices(simulator_train, filtered_signal_train, bin = bin)
    X_test, _ = make_matrices(None, filtered_signal_test, bin = bin)

    # convert to frquency
    X_train = convert_to_frequency(X_train)
    X_test = convert_to_frequency(X_test)

    # normalize data 
    X_train = X_train / np.max(X_train, axis = 1).reshape(-1, 1)
    X_test = X_test / np.max(X_test, axis = 1).reshape(-1, 1)
    y_max = np.max(y_train)
    y_train = y_train / y_max if y_max != 0 else y_train

    # initialize the regressor 
    regressor = SVR(kernel = "linear")

    # train the regressor
    regressor.fit(X_train, y_train) 

    # predict the test data
    y_pred_test = regressor.predict(X_test) 

    return y_pred_test * y_max