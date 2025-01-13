from src.model.baseline_model import count_caps_baseline
from src.model.wavelet_model import count_caps_wavelet
from src.model.svm_model import count_caps_svm
from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter 
from multiprocessing import Pool
import numpy as np 
import pickle 
import os 


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
    prefix = "../../../../../../../../work3/s194329/"
    os.makedirs(prefix + "results", exist_ok=True)
    
    for i in range(noise_params.shape[0]): 
        os.makedirs(f"{prefix}results/noise_config_{i}", exist_ok=True)


def count_true_caps(simulator : SimulateData) -> np.ndarray:
    """ Function that counts the true number of CAPs in the signal """

    # utility function to count the length of a CAP
    CAP_length = lambda x: len(x) if type(x) == list else 0

    # allocate memory for the counts
    all_true_counts = np.zeros((simulator.num_channels, int(simulator.duration * simulator.stim_freq)))

    for channel in range(simulator.num_channels):
        for bin_idx in range(int(simulator.duration * simulator.stim_freq)):
            all_true_counts[channel, bin_idx] = CAP_length(simulator.CAP_indices[bin_idx][channel]) 

    return all_true_counts

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
    
    # construct save name 
    save_name = f"../../../../../../../../work3/s194329/results/noise_config_{noise_dist}/snr_{int(snr*10)}_count_{count}.pkl"
    # save_name = f"../../results/baseline/noise_config_{noise_dist}/snr_{int(snr*10)}_count_{count}.pkl"

    if os.path.exists(save_name): 
        print(f"Skipping save_name : {save_name}")
        return

    # make simulated data
    seed = hash(filename) % (2**32)
    simulator = SimulateData(snr, [pli, hz_500, white, high_freq], CAP_dist="uniform", seed = seed)
    simulator.construct_signal()

    # filter signal 
    print("filter signal")
    filtered_signal = filter(simulator.signal)

    # count CAPS using different methods 
    print("baseline and wavelet")
    estimated_caps_baseline = count_caps_baseline(simulator.signal, filtered_signal)
    estimated_caps_wavelet = count_caps_wavelet(simulator.signal, filtered_signal)

    # make new instance of simulator for SVM to train 
    print("svm")
    simulator_train = SimulateData(snr, [pli, hz_500, white, high_freq], CAP_dist="uniform", seed = seed-1)
    simulator_train.construct_signal()
    filtered_signal_train = filter(simulator_train.signal)
    estimated_caps_svm = count_caps_svm(simulator_train, filtered_signal_train, filtered_signal)

    # count true CAPS
    true_caps = count_true_caps(simulator)

    # save to dictionary 
    d = {}
    d['estimated_baseline'] = estimated_caps_baseline 
    d['estimated_wavelet'] = estimated_caps_wavelet 
    d['estimated_svm'] = estimated_caps_svm  
    d['true'] = true_caps
    d['obs_signal'] = simulator.signal 
    d['true_signal'] = simulator.true_signal
    d['filtered_signal'] = filtered_signal

    # save files 
    print(f"saving {save_name}")
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
    all_snrs = np.array(([0.1, 0.5, 1, 1.5, 2]))
    noise_params = np.array([[200, 1, 10, 20], 
                            [300, 1, 10, 20], 
                            [200, 5, 10, 20], 
                            [200, 1, 30, 20],
                            [200, 1, 10, 40]])
    n_repeats = 5 
    
    # create folders to save results to 
    create_folders(noise_params)

    # change directory
    os.chdir("src/data")

    # run the counting 
    count_all(all_snrs, noise_params, n_repeats)

