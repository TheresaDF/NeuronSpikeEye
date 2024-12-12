import os 
import pickle 
from multiprocessing import Pool
import numpy as np 
from baseline_model import count_caps
from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter 



def generate_file_names(snsr : np.ndarray, noise : np.ndarray) -> list:  
    all_filenames = []
    for n in noise: 
        for s in snsr: 
            for count in range(10): 
                filename = f"sim_" + str(int(s*10)) + "_" + str(n[0]) + "_" + str(n[1]) + "_" + str(n[2]) + "_" + str(n[3]) + "_" + str(count)
                all_filenames.append(filename)

    return all_filenames

def create_folders(noise_params : np.ndarray):
    for i in range(noise_params.shape[0]): 
        os.makedirs(f"results/baseline/noise_config_{i}")


def counter(filename : str, noise_params : int, count : int) -> None: 
    # unpack parameters from filename 

    filename_parts = filename.split("_")
    snr = filename_parts[1] / 10 
    pli = filename_parts[2]
    hz_500 = filename_parts[3]
    white = filename_parts[4]
    high_freq = filename_parts[5]  
    
    # make simulated data
    simulator = SimulateData(snr, [pli, hz_500, white, high_freq], CAP_freq = 10, CAP_dist="lognormal")
    simulator.construct_signal()

    # filter signal 
    filtered_signal = filter(simulator.signal)

    # count CAPS 
    estimated_caps, true_caps = count_caps(simulator, filtered_signal)

    # save to dictionary 
    d = {}
    d['estimated'] = estimated_caps
    d['true'] = true_caps
    d['obs_signal'] = simulator.signal 
    d['true_signal'] = simulator.true_signal
    d['filtered_signal'] = filtered_signal

    # construct save name 
    save_name = f"results/baseline/noise_config_{noise_params}/snr_{int(snr*10)}_count_{count}.pkl"

    # save files 
    with open(save_name) as output_file: 
        pickle.dump(d, output_file)
    output_file.close()
    
def count_all(all_snrs : np.ndarray, noise : np.ndarray):
    # get filenames 
    all_filenames = generate_file_names(all_snrs, noise)

    # start processing all files 
    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(counter, all_filenames) 
    

if __name__ == "__main__":
    all_snrs = np.linspace(0.1, 1.9, num=18, endpoint=True)
    noise_params = np.array([[200, 150, 10, 1], 
                            [300, 200, 10, 1], 
                            [200, 150, 10, 1], 
                            [200, 150, 30, 1],
                            [200, 150, 10, 5]])
    
    # create folders to save results to 
    create_folders(noise_params)

    # run the counting 
    count_all(all_snrs, noise_params)
