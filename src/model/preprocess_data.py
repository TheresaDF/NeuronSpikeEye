from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter 
from multiprocessing import Pool
import numpy as np 
import pickle 
import os 

def generate_inputs(data_type : str, snsr : np.ndarray, noise : np.ndarray, n_repeats : int = 10) -> list:  
    all_inputs = []
    for n_count, n in enumerate(noise): 
        for s in snsr: 
            for count in range(n_repeats): 
                filename = f"{data_type}_" + str(int(s*10)) + "_" + str(n[0]) + "_" + str(n[1]) + "_" + str(n[2]) + "_" + str(n[3]) + "_" + str(count)
                n_dist = n_count 
                all_inputs.append((filename, n_dist))

    return all_inputs 


def create_folders(data_type : str, noise_params : np.ndarray):
    prefix = "../../../../../../../../work3/s194329/"  
    name = "results_synthetic_" + data_type
    os.makedirs(prefix + name, exist_ok=True)
    
    for i in range(noise_params.shape[0]): 
        os.makedirs(f"{prefix}{name}/filtered_data", exist_ok=True)

def filter_single(args : tuple[str, int]) -> None: 

    # unpack arguments 
    filename, noise_dist = args 
    
    # unpack parameters from filename 
    filename_parts = filename.split("_")
    data_type = filename_parts[0]
    snr = float(filename_parts[1]) / 10 
    pli = int(filename_parts[2])
    hz_500 = int(filename_parts[3])
    white = int(filename_parts[4])
    high_freq = int(filename_parts[5])
    count = int(filename_parts[6])
    
    # construct save name 
    save_name = f"../../../../../../../../work3/s194329/results_synthetic_{data_type}/filtered_data/snr_{int(snr*10)}_count_{count}.pkl"

    if os.path.exists(save_name): 
        print(f"Skipping save_name : {save_name}")
        return

    # determine if data should be binned 
    to_bin = False if data_type == "spon" else True

    # make simulated data
    seed = hash(filename) % (2**32)
    CAP_dist = "uniform" if data_type == "stim" else None 
    simulator = SimulateData(snr, [pli, hz_500, white, high_freq], CAP_dist=CAP_dist, seed = seed)    
    simulator.construct_signal()

    # filter signal 
    print("filter signal")
    filtered_signal, _ = filter(simulator.signal, bin = to_bin)

    d = {}
    d['filt_signal'] = filtered_signal
    
    print(f"saving {save_name}")
    with open(save_name, 'wb') as output_file: 
        pickle.dump(d, output_file)
    output_file.close()

def preprocess(data_type : str, all_snrs : np.ndarray, noise_params : np.ndarray, n_repeats : int): 
    inputs = generate_inputs(data_type, all_snrs, noise_params, n_repeats)

    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(filter_single, inputs) 

if __name__ == "__main__":
    all_snrs = np.array([0, 0.1, 1, 2, 3, 4])
    noise_params = np.array([[200, 1, 10, 20]]) 
                            #[45, 0, 10, 20]
                            # [300, 1, 10, 20], 
                            # [200, 50, 10, 20], 
                            # [200, 1, 30, 20],
                            # [200, 1, 10, 40]])
    n_repeats = 30 
    data_type = "stim"
     
    # create folders to save results to 
    create_folders(data_type, noise_params)

    # change directory
    os.chdir("src/data")

    # run the counting 
    preprocess(data_type, all_snrs, noise_params, n_repeats)

