import os 
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from src.model.baseline_model import count_caps_baseline
from src.model.wavelet_model import count_caps_wavelet
from src.model.svm_model import count_caps_svm
from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter 
from multiprocessing import Pool
import numpy as np 
import pickle 


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
        os.makedirs(f"{prefix}{name}/noise_config_{i}", exist_ok=True)


def count_true_caps(simulator : SimulateData) -> np.ndarray:
    """ Function that counts the true number of CAPs in the signal """

    # utility function to count the length of a CAP
    CAP_length = lambda x: len(x) if type(x) == list else 0

    # allocate memory for the counts
    all_true_counts = np.zeros((simulator.num_channels, int(simulator.duration * simulator.stim_freq)))

    if simulator.CAP_indices is None:
        return all_true_counts
    else: 
        for channel in range(simulator.num_channels):
            for bin_idx in range(int(simulator.duration * simulator.stim_freq)):
                all_true_counts[channel, bin_idx] = CAP_length(simulator.CAP_indices[bin_idx][channel]) 

        return all_true_counts

def counter(args : tuple[str, int]) -> None: 

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
    save_name = f"../../../../../../../../work3/s194329/results_synthetic_{data_type}/noise_config_{noise_dist}/snr_{int(snr*10)}_count_{count}.pkl"

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

    # count CAPS using different methods 
    print("baseline and wavelet")
    estimated_caps_baseline = count_caps_baseline(simulator.signal, filtered_signal, bin = to_bin)
    estimated_caps_wavelet = count_caps_wavelet(simulator.signal, filtered_signal, bin = to_bin)

    # make new instance of simulator for SVM to train 
    print("svm")
    simulator_train = SimulateData(snr, [pli, hz_500, white, high_freq], CAP_dist=CAP_dist, seed = seed-1, num_channels = 32*3)
        
    simulator_train.construct_signal()
    filtered_signal_train, _ = filter(simulator_train.signal, max_count = 30, bin = to_bin)
    estimated_caps_svm = count_caps_svm(simulator_train, filtered_signal_train, filtered_signal, bin = to_bin)

    # count true CAPS
    true_caps = count_true_caps(simulator)

    # get mean for mean predictor
    true_caps_train = count_true_caps(simulator_train)
    mean_predict = np.mean(np.sum(true_caps_train, axis = 0)) * np.ones(simulator_train.num_channels)

    # save to dictionary 
    d = {}
    d['estimated_baseline'] = estimated_caps_baseline 
    d['estimated_wavelet'] = estimated_caps_wavelet 
    d['estimated_svm'] = estimated_caps_svm  
    d['estimated_mean_predict'] = mean_predict
    d['true'] = true_caps

    # save files 
    print(f"saving {save_name}")
    with open(save_name, 'wb') as output_file: 
        pickle.dump(d, output_file)
    output_file.close()
    
def count_all(data_type : str, all_snrs : np.ndarray, noise : np.ndarray, n_repeats : int):
    # get filenames 
    inputs = generate_inputs(data_type, all_snrs, noise, n_repeats)

    # start processing all files 
    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(counter, inputs) 
    

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
    count_all(data_type, all_snrs, noise_params, n_repeats)