from src.data.dataset_utils import read_data, read_ramp
from src.data.precprocess_utils import filter 
import numpy as np
import os


def pre_process() -> None:
    # make directories for 2D and 3D 
    os.makedirs("data/preprocessed", exist_ok = True)
    os.makedirs("data/preprocessed/2D", exist_ok = True)
    os.makedirs("data/preprocessed/3D", exist_ok = True)
    for eye in range(6):
        os.makedirs(f"data/preprocessed/2D/Eye {eye+1}", exist_ok = True)
        os.makedirs(f"data/preprocessed/3D/Eye {eye+1}", exist_ok = True)

    # loop over data 
    print("Processing 2D and 3D data")
    for electrode in ["2D", "3D"]:
        for eye in range(6):
            print(f"Processing Eye {eye+1} for {electrode}")
            # read data 
            data_stim, data_ttx, data_spon = read_data(eye, electrode)

            # preprocess data 
            data_stim_processed = filter(data_stim)
            data_ttx_processed = filter(data_ttx)
            data_spon_processed = filter(data_spon)

            # save result 
            np.save(f"data/preprocessed/{electrode}/Eye {eye+1}/stim.npy", data_stim_processed)
            np.save(f"data/preprocessed/{electrode}/Eye {eye+1}/ttx.npy", data_ttx_processed)
            np.save(f"data/preprocessed/{electrode}/Eye {eye+1}/spon.npy", data_spon_processed)

    # for ramp data 
    os.makedirs("data/preprocessed/Ramp", exist_ok = True)
    for eye in range(6):
        os.makedirs(f"data/preprocessed/Eye {eye + 1}/Ramp data/", exist_ok = True)

    # loop over data
    print("Processing Ramp data")
    for eye in range(6):
        for ramp in range(14):
            print(f"Processing Eye {eye+1} for Ramp")
            # read data
            data_stim = read_ramp(eye, ramp)

            # preprocess data
            data_stim_processed = filter(data_stim)

            # save result
            np.save(f"data/preprocessed/Ramp data/Eye {eye+1}/ramp_file{ramp+1}.npy", data_stim_processed)

if __name__ == "__main__":
    pre_process()
