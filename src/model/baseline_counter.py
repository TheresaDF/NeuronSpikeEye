import os 
import pickle 
from multiprocessing import Pool
import numpy as np 
from baseline_model import count_caps
from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter 



def generate_inputs(snsr : np.ndarray, noise : np.ndarray, n_repeats : int = 10) -> list:  
    all_inputs = []
    for n_count, n in enumerate(noise): 
        for s in snsr: 
            for count in range(n_repeats): 
                filename = f"sim_" + str(int(s*10)) + "_" + str(n[0]) + "_" + str(n[1]) + "_" + str(n[2]) + "_" + str(n[3]) + "_" + str(count)
                n_dist = n_count 
                all_inputs.append((filename, n_dist))

    return all_inputs 


def create_folders(noise_params : np.ndarray):
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/baseline", exist_ok=True)
    for i in range(noise_params.shape[0]): 
        os.makedirs(f"results/baseline/noise_config_{i}")


def counter(args : tuple[str, int, int]) -> None: 

    # unpack arguments 
    filename, noise_dist = args 
    
    # unpack parameters from filename 
    filename_parts = filename.split("_")
    snr = float(filename_parts[1]) / 10 
    pli = int(filename_parts[2])
    hz_500 = int(filename_parts[3])
    white = int(filename_parts[4])
    high_freq = int(filename_parts[5])
    count = int(filename_parts[6])
    
    # make simulated data
    seed = np.random.seed(hash(filename) % (2**32))
    simulator = SimulateData(snr, [pli, hz_500, white, high_freq], CAP_freq = 4, CAP_dist="lognormal", seed = seed)
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
    save_name = f"../../results/baseline/noise_config_{noise_dist}/snr_{int(snr*10)}_count_{count}.pkl"

    # save files 
    with open(save_name, 'wb') as output_file: 
        pickle.dump(d, output_file)
    output_file.close()
    
def count_all(all_snrs : np.ndarray, noise : np.ndarray, n_repeats : int):
    # get filenames 
    inputs = generate_inputs(all_snrs, noise, n_repeats)
    

    # start processing all files 
    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(counter, inputs) 
    
    # pool = Pool(processes=2)
    # pool.map(counter, inputs)

if __name__ == "__main__":
    all_snrs = np.linspace(0.1, 1.9, num=18, endpoint=True)
    noise_params = np.array([[200, 1, 10, 20], 
                            [300, 5, 10, 20], 
                            [200, 1, 10, 20], 
                            [200, 1, 30, 20],
                            [200, 1, 10, 40]])
    n_repeats = 10 
    
    # create folders to save results to 
    create_folders(noise_params)

    # change directory
    os.chdir("src/data")

    # run the counting 
    count_all(all_snrs, noise_params, n_repeats)

