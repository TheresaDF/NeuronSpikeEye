import neo 
import glob
import os 
import numpy as np 
from scipy.signal import find_peaks, fftconvolve
from scipy.fft import fft, fftfreq, ifft



def read_ns5_file(filename : str) -> tuple[np.ndarray, np.ndarray]:
    """ Function to read files of .ns5 format and return the data and time values."""
    reader = neo.io.BlackrockIO(filename = filename, verbose = True)
    times = reader.read_block(0).segments[0].analogsignals[0].times 
    data = reader.read_block(0).segments[0].analogsignals[0].magnitude

    return times, data

def read_data(eye: int, data_type : str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """ Function reads all raw files for certain eye and type"""
    
    # get path to data files 
    paths = glob.glob(os.path.join("data/raw/", data_type, "Eye " + str(eye+1), "*.ns5"))
    
    # read data
    _, data_stim = read_ns5_file(paths[0])
    _, data_ttx = read_ns5_file(paths[1])
    _, data_spon = read_ns5_file(paths[2])

    return data_stim, data_ttx, data_spon

def read_ramp(eye : int, ramp : int) -> np.ndarray:
    """ Function reads ramp data for certain eye"""
    
    # get path to data files 
    paths = sorted(glob.glob(os.path.join("data/raw/Ramp data/Eye " + str(eye+1), "*.ns5")), key = len)

    # read data
    _, data = read_ns5_file(paths[ramp])

    return data

def get_spike(data : np.ndarray) -> np.ndarray:
    """Get spike from 'clean' 2D data """   
    # get spike from 2D data where they are pronounced 
    
    stim_peak = np.argmax(data)
    spike = data[stim_peak-40:stim_peak+100] 

    # mirror peak and zero pad 
    peak_mirror = np.r_[np.flip(spike), np.zeros(3000-140)]

    return peak_mirror   

def locate_peaks(data_3d : np.ndarray, spike : np.ndarray) -> np.ndarray:
    """Function locates peaks in 3D by using matched filtering"""
    
    # repeat spike 100 times
    filter = np.array([spike for _ in range(100)]).reshape(-1)

    # convolve with the 3D data
    conv = fftconvolve(data_3d, filter / np.sum(filter), mode = "same")

    # find peaks 
    dist = 2200 
    peaks = []
    while len(peaks) != 100:
        peaks, _ = find_peaks(conv, distance=dist, height = 200)
        dist += 10 

    # there should be a 100 peaks
    assert len(peaks) == 100 
    return peaks 

def bin_spon(spon_data: np.ndarray) -> np.ndarray:
    """Function bins spontaneous data"""

    # number of datapoints if we didvide into a 100 segments 
    n = len(spon_data)
    length = int(n / 100 )

    # divide in segments 
    segments = np.zeros((length, 100))
    for i in range(100):
        segments[:, i] = spon_data[i*length:(i+1)*length]

    return segments

def bin_data(data : np.ndarray, spike = np.ndarray) -> np.ndarray:
    """Function bins data based on peak locations"""

    # detect peaks             
    peaks = locate_peaks(data, spike)
    
    # there should be a 100 peaks 
    assert len(peaks) == 100 

    # break up into segments based on spike location 
    segments = np.zeros((2400, 100))
    for i in range(100):    
        if (i == 99 ) & (len(data) - peaks[i] - 300 < 2400):
            points_left = len(data) - peaks[i] - 300 
            segments[:points_left, i] = data[peaks[i]+300:peaks[i]+300+points_left]
        else: 
            segments[:, i] = data[peaks[i]+300:peaks[i]+2700]

    return segments 


def add_to_dictionary(eye : int, d : dict[str, str]) -> dict[str, str]:
    """Function initializes dictionaries to save data"""

    # add eye to dict 
    if len(d) == 0: 
        d['2D'] = {'Eye ' + str(eye+1) :  np.array([{} for _ in range(32)])}
        d['3D'] = {'Eye ' + str(eye+1) :  np.array([{} for _ in range(32)])}
    else: 
        d['2D']['Eye ' + str(eye + 1)] = np.array([{} for _ in range(32)])
        d['3D']['Eye ' + str(eye + 1)] = np.array([{} for _ in range(32)])

    return d 



from scipy.signal import iirnotch, lfilter
def notch_filter(x, f0, fs, Q = 30):
    b, a = iirnotch(f0 / (fs / 2), Q)
    y = lfilter(b, a, x)
    
    return y

def apply_notch_filter(data, f0, num, sample_rate = 3*1e4, Q = 30):
    """
    Apply notch filter to data
    """
    filtered_data = data.copy()
    for i in range(num): 
        filtered_data = notch_filter(filtered_data, (i+1) * f0, sample_rate, Q = Q)

    return filtered_data