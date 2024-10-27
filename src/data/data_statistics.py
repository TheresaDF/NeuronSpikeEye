from src.data.utils import read_ns5_file
import statsmodels.tsa.stattools as ts 
from scipy.signal import welch
import numpy as np 
import os 
import glob 

def get_ramp(eye : int, ramp : int) -> np.ndarray:
    """ Function reads ramp data for certain eye"""
    
    # get path to data files 
    paths = glob.glob(os.path.join("data/raw/Ramp data/Eye " + str(eye+1), "*.ns5"))

    # read data
    _, data = read_ns5_file(paths[ramp])

    return data

def get_eye(eye: int, data_type : str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Function reads all raw files for certain eye and type"""
    
    # get path to data files 
    paths = glob.glob(os.path.join("data/raw/", data_type, "Eye " + str(eye+1), "*.ns5"))
    
    # read data
    _, data_stim = read_ns5_file(paths[0])
    _, data_ttx = read_ns5_file(paths[1])
    _, data_spon = read_ns5_file(paths[2])

    return data_stim, data_ttx, data_spon

def get_simulated_data(path : str = "data/simulated/10_30_lognormal.npy") -> np.ndarray:
    """ Function returns simulated data"""

    return np.load(path)


def get_df(data: np.ndarray) -> tuple[float, float]:
    """ Function calculates dominant frequency of data"""

    num_channels = data.shape[1]
    max_freq = np.zeros(num_channels)

    for channel in range(num_channels):
        p, fxx = welch(data[:, channel], fs = 30000, nperseg = 512)
        max_freq[channel] = p[np.argmax(fxx)]

    return np.mean(max_freq), np.std(max_freq)

def get_acf(data : np.ndarray):
    """ Function calculates autocorrelation function of data"""

    num_channels = data.shape[1]
    acf = np.zeros((num_channels, 3))
    for channel in range(num_channels):
        acf[channel] = ts.acf(data[:, channel], nlags=2)

    return np.mean(acf, axis = 0)

def statistics(data: np.ndarray) -> dict:
    """ Function calculates mean, std, min and max of data"""
    
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "IQR": np.percentile(data, 75) - np.percentile(data, 25),
        "median": np.median(data),
        "ACF" : np.round(get_acf(data), 2),
        "DF" : get_df(data)
    }

def get_statistics_ramp(eye: int, ramp: int) -> dict:
    """ Function calculates statistics for all data types of certain eye"""

    data_ramp = get_ramp(eye, ramp)

    return {
        "ramp": statistics(data_ramp)
    }

def get_statistics(eye: int, data_type: str) -> dict:
    """ Function calculates statistics for all data types of certain eye"""

    data_stim, data_ttx, data_spon = get_eye(eye, data_type)

    return {
        "stim": statistics(data_stim),
        "ttx": statistics(data_ttx),
        "spon": statistics(data_spon)
    }

def get_statistics_simulated() -> dict:
    """ Function calculates statistics for all data types of certain eye"""

    data_simulated = get_simulated_data()

    return {
        "simulated": statistics(data_simulated)
    }

def get_all_statistics(electrode : str, data_type : str, ramp : int) -> dict:
    """ Function calculates statistics for all eyes and data types"""
    if data_type == "ramp":
        return {
            "ramp1": get_statistics_ramp(0, ramp),
            "ramp2": get_statistics_ramp(1, ramp),
            "ramp3": get_statistics_ramp(2, ramp),
            "ramp4": get_statistics_ramp(3, ramp),
            "ramp5": get_statistics_ramp(4, ramp),
            "ramp6": get_statistics_ramp(5, ramp)
        }
    elif data_type == "simulated":
        return { 
            "sim1": get_statistics_simulated()
        }
    else: 
        return {
            "eye1": get_statistics(0, electrode),
            "eye2": get_statistics(1, electrode),
            "eye3": get_statistics(2, electrode),
            "eye4": get_statistics(3, electrode),
            "eye5": get_statistics(4, electrode),
            "eye6": get_statistics(5, electrode),
        }

def print_statistics(electrode : str = "2D", data_type : str = "spon", ramp : int = 0):
    """ Function prints all statistics"""

    stats = get_all_statistics(electrode, data_type, ramp)

    for eye in stats:
        if data_type == "ramp":
            print(f"Eye {eye[4]}")
        else: 
            print(f"Eye {eye[3]}")
    
        print(f"Data type: {data_type}")
        for stat in stats[eye][data_type]:
            print(f"{stat}: {stats[eye][data_type][stat]}")
        print("\n")

if __name__ == "__main__":
    # # # # # Choose parameters # # # # #
    electrode = "2D" # "2D" or "3D"
    data_type = "stim" # "stim", "ttx" or "spon",  "ramp", "simulated"
    ramp = None # 0 to 13 (only used if data_type = "ramp")

    print_statistics(electrode = electrode, data_type=data_type, ramp = ramp)