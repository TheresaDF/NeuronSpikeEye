from src.data.dataset_utils import read_data, read_ramp
from src.data.preprocess_utils import filter 
import numpy as np
import os
from multiprocessing import Pool

def create_directories() -> None:
    os.makedirs("data/preprocessed", exist_ok=True)
    os.makedirs("data/preprocessed/2D", exist_ok=True)
    os.makedirs("data/preprocessed/3D", exist_ok=True)
    for eye in range(6):
        os.makedirs(f"data/preprocessed/2D/Eye {eye+1}", exist_ok=True)
        os.makedirs(f"data/preprocessed/3D/Eye {eye+1}", exist_ok=True)
    os.makedirs("data/preprocessed/Ramp data", exist_ok=True)
    for eye in range(6):
        os.makedirs(f"data/preprocessed/Ramp data/Eye {eye+1}/", exist_ok=True)

def process_eye_electrode(args : tuple[int, str]) -> None:
    eye, electrode = args[0], args[1]
    print(f"Processing Eye {eye+1} for {electrode}")

    # Read data
    data_stim, data_ttx, data_spon = read_data(eye, electrode)

    # Preprocess data
    data_stim_processed = filter(data_stim)
    data_ttx_processed = filter(data_ttx)
    data_spon_processed = filter(data_spon)

    # Save results
    np.save(f"data/preprocessed/{electrode}/Eye {eye+1}/stim.npy", data_stim_processed)
    np.save(f"data/preprocessed/{electrode}/Eye {eye+1}/ttx.npy", data_ttx_processed)
    np.save(f"data/preprocessed/{electrode}/Eye {eye+1}/spon.npy", data_spon_processed)

def process_eye_ramp(args : tuple[int, int]) -> None:
    eye, ramp = args[0], args[1]
    print(f"Processing Eye {eye+1} for Ramp {ramp+1}")

    # Read data
    data_stim = read_ramp(eye, ramp)

    # Preprocess data
    data_stim_processed = filter(data_stim)

    # Save results
    np.save(f"data/preprocessed/Ramp data/Eye {eye+1}/ramp_file{ramp+1}.npy", data_stim_processed)

def pre_process() -> None:
    # Create directories
    create_directories()

    # Prepare tasks
    print("Processing 2D and 3D data")
    tasks_2d_3d = [(eye, electrode) for electrode in ["2D", "3D"] for eye in range(6)]

    print("Processing Ramp data")
    tasks_ramp = [(eye, ramp) for eye in range(6) for ramp in range(14)]

    # Run the preprocessing in parallel
    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(process_eye_electrode, tasks_2d_3d)

        # Process Ramp data
        result = pool.map(process_eye_ramp, tasks_ramp)

if __name__ == "__main__":
    pre_process()
