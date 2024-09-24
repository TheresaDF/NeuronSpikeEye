import os 
import glob 
import pickle 
import numpy as np 
from utils import *
from tqdm import tqdm 
from scipy.signal import find_peaks, fftconvolve


def read_data(eye: int, data_type : str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """ Function reads all raw files for certain eye and type"""
    
    # get path to data files 
    paths = glob.glob(os.path.join("data/raw/", data_type, "Eye " + str(eye+1), "*.ns5"))
    
    # read data
    _, data_stim = read_ns5_file(paths[0])
    _, data_ttx = read_ns5_file(paths[1])
    _, data_spon = read_ns5_file(paths[2])

    return data_stim, data_ttx, data_spon

def get_spike(data : np.ndarray) -> np.ndarray:
    """Get spike from 'clean' 2D data """   
    # get spike from 2D data where they are pronounced 
    
    stim_peak = np.argmax(data)
    spike = data[stim_peak-33:stim_peak+100] 

    return spike 

def locate_peaks_3d(data_3d : np.ndarray, spike : np.ndarray) -> np.ndarray:
    """Function locates peaks in 3D by using matched filtering"""
    
    # repeat spike 100 times
    filter = np.array([spike for _ in range(100)]).reshape(-1)

    # convolve with the 3D data
    conv = fftconvolve(data_3d, filter / np.sum(filter), mode = "same")

    # find peaks 
    peaks, _ = find_peaks(conv, distance=2200, height = 200)

    # there should be a 100 peaks 
    assert len(peaks) == 100 
    return peaks 

def bin_spon(spon_data: np.ndarray) -> np.ndarray:
    """Function bins spontaneous data"""

    # number of datapoints if we didvide into a 100 segments 
    n = len(spon_data)
    length = n / 100 

    # divide in segments 
    segments = np.zeros((length, 100))
    for i in range(100):
        segments[:, i] = spon_data[i*100:(i+1)*100]

    return segments

def bin_data(data : np.ndarray, data_type : str, spike = None) -> np.ndarray:
    """Function bins data based on peak locations"""

    # detect peaks 
    if data_type == "2D":
        peaks, _ = find_peaks(data, distance = 2200, height = 300)
    else:
        peaks = locate_peaks_3d(data, spike)
    
    # there should be a 100 peaks 
    print(len(peaks))
    assert len(peaks) == 100 

    # break up into segments based on spike location 
    segments = np.zeros((3000-33-100, 100))
    for i in range(100):
        segments[:, i] = data[peaks[i]+100:peaks[i]+2967]

    return segments 

def add_to_dictionary(eye : int, d : dict[str, str]) -> dict[str, str]:
    """Function initializes dictionaries to save data"""

    # add eye to dict 
    d['2D'] = {'Eye ' + str(eye+1) :  np.array([{} for _ in range(32)])}
    d['3D'] = {'Eye ' + str(eye+1) :  np.array([{} for _ in range(32)])}

    return d 


def pre_process() -> None:
    """Function bins all data according to locations of stimulation artefacts"""
    
    # initialize dictionaries for stim, TTX and spon
    d_stim = {}
    d_ttx = {}
    d_spon = {}


    # loop over eyes 
    for eye in tqdm(range(6)):
        # add eye to dictionaries 
        d_stim = add_to_dictionary(eye, d_stim)
        d_ttx = add_to_dictionary(eye, d_ttx)
        d_spon = add_to_dictionary(eye, d_spon)

        # read 2D data 
        stim_2, ttx_2, spon_2 = read_data(eye, "2D")

        # read 3D data 
        stim_3, ttx_3, spon_3 = read_data(eye, "3D")

        # loop over channels
        for channel in range(32):
            print(f"Channel {channel+1}/32")
            # bin 2D data 
            segments_stim = bin_data(stim_2[:, channel], '2D')
            segments_ttx = bin_data(ttx_2[:, channel], '2D')
            segments_spon = bin_spon(spon_2[:, channel], '2D')

            # save to dictionary 
            d_stim['2D']['Eye ' + str(eye+1)][channel] = segments_stim
            d_ttx['2D']['Eye ' + str(eye+1)][channel] = segments_ttx
            d_spon['2D']['Eye ' + str(eye+1)][channel] = segments_spon

            # read spike from 2D data
            spike = get_spike(stim_2[:, channel])

            # bin 3D data 
            segments_stim = bin_data(stim_3[:, channel], '3D', spike = spike)
            segments_ttx = bin_data(ttx_3[:, channel], '3D', spike = spike)
            segments_spon = bin_spon(spon_3[:, channel], '3D', spike = spike)

            # save to dictionary 
            d_stim['3D']['Eye ' + str(eye+1)][channel] = segments_stim
            d_ttx['3D']['Eye ' + str(eye+1)][channel] = segments_ttx
            d_spon['3D']['Eye ' + str(eye+1)][channel] = segments_spon

    # save results 
    with open('data/processed/stimulation.pkl', 'wb') as f: pickle.dump(d_stim, f); f.close()
    with open('data/processed/spontaneous.pkl', 'wb') as f: pickle.dump(d_spon, f); f.close()
    with open('data/processed/ttx.pkl', 'wb') as f: pickle.dump(d_ttx, f); f.close()


if __name__ == "__main__":
    pre_process()