from scipy.io import loadmat, savemat
from utils import read_ns5_file
import numpy as np
from tqdm import tqdm 
import os 
import glob 

# file to save data as mat files 

def get_file(electrode : str, eye : int, data_type : int) -> np.ndarray:
    """ Function to read raw file
    datatype: 0 = stim, 1 = TTX, 2 = spontaneous 
    """
    
    path = "data/raw/" + electrode + "/Eye " + str(eye+1) 
    filename = glob.glob(path + "/*.ns5")[data_type]
    _, data = read_ns5_file(filename)

    return data if len(data) == 300300 else data[:300300]

def get_single_ramp(eye : int, ramp : int) -> np.ndarray:
    """ Function to read ramp file"""

    path = "data/raw/Ramp data/Eye " + str(eye+1)
    filename = glob.glob(path + "/*.ns5")[ramp]
    _, data = read_ns5_file(filename)

    return data  

def get_all_ramps(eye : int) -> np.ndarray:
    """ Function to read all ramp data for a iven eye"""

    all_ramps = np.zeros((13, 300300, 32))
    for ramp in range(13):
        data = get_single_ramp(eye, ramp)
        all_ramps[ramp] = data 

    return all_ramps


def save_to_mat() -> None:
    """ Function to save all the data to mat files"""

    data_types = ["2D", "3D"]

    # save the 2D and 3D data 
    print("Save 2D and 3D data")
    for dt in tqdm(data_types):
        stim_all, TTX_all, spon_all = np.zeros((6, 300300, 32)), np.zeros((6, 300300, 32)), np.zeros((6, 300300, 32))
        for eye in range(6):
            stim = get_file(dt, eye, 0)
            TTX = get_file(dt, eye, 1)
            spon = get_file(dt, eye, 2)

            stim_all[eye] = stim 
            TTX_all[eye] = TTX 
            spon_all[eye] = spon 

        savemat("data/mat_files/raw/" + dt + "_stim.mat", {"stim" + dt: stim_all})
        savemat("data/mat_files/raw/" + dt + "_TTX.mat", {"TTX" + dt: TTX_all}) 
        savemat("data/mat_files/raw/" + dt + "_spon.mat", {"spon" + dt: spon_all})
    
    
    # save the ramp data 
    print("save ramp data")
    for eye in tqdm(range(6)):
        all_ramps = get_all_ramps(eye)
        save_name = "data/mat_files/raw/ramp_eye" + str(eye+1) + ".mat"

        savemat(save_name, {"ramp_eye" + str(eye+1) : all_ramps})
                

if __name__ == "__main__":
    save_to_mat()


