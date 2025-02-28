from src.data.create_simulated_data import SimulateData
from sklearn.model_selection import train_test_split
from src.data.preprocess_utils import bin_data
from scipy.signal import find_peaks, welch 
from skopt import gp_minimize
from skopt.space import Real 
from sklearn.svm import SVR 
import numpy as np


CAP_length = lambda x: len(x) if type(x) == list else 0

def make_matrices(simulator : SimulateData, filtered_signal : np.ndarray, duration : int = 10, stim_freq : int = 10, bin = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Make matrices for training and testing
    """
    # get the number of caps 
    num_channels = filtered_signal.shape[1]
    if simulator is not None and getattr(simulator, "CAP_indices", None) is not None: 
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

def find_hyper_parameters(X_optim, y_optim): 
    # Objective function to minimize
    def objective(params):
        C, epsilon = params  # Extract parameters from the optimization

        # Define the custom kernel wrapper for SVR
        svr = SVR(kernel='linear', C=C, epsilon=epsilon)
        
        # Split the data 
        X_train, X_val, y_train, y_val = train_test_split(X_optim, y_optim, test_size=0.2, random_state=42)
        
        # Fit and predict
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_val)
        
        # Calculate RMSE on the validation set
        rmse_val = np.sqrt(np.mean((y_val - y_pred) ** 2))
        
        return rmse_val

    # Define the search space 
    search_space = [
        Real(1e-2, 1e3, name="C"),             # Regularization parameter
        Real(1e-3, 1e1, name="epsilon"),       # Tube size in regression
    ]

    # Run Bayesian optimization
    results = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=30,  # Number of function evaluations
        random_state=42
    )

    # Best hyperparameters and RMSE
    C = results.x[0]
    epsilon = results.x[1]

    return C, epsilon

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

    # find optimal parameters 
    C, epsilon = find_hyper_parameters(X_train, y_train)

    # initialize the regressor 
    regressor = SVR(kernel = "linear", C = C, epsilon = epsilon)

    # train the regressor
    regressor.fit(X_train, y_train) 

    # predict the test data
    y_pred_test = regressor.predict(X_test) 

    return y_pred_test * y_max