from src.model.baseline_model import count_caps_baseline
from src.model.wavelet_model import count_caps_wavelet
from src.model.svm_model import count_caps_svm
from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter, read_ns5_file
from src.model.counter import count_true_caps
from multiprocessing import Pool
import numpy as np 
import pickle 
import glob 
import os 


def create_folders(data_type : str) -> None:
    prefix = "../../../../../../../../work3/s194329/"
    os.makedirs(prefix + f"results_real_{data_type}", exist_ok=True)

def counter(args : tuple[str, str]) -> None: 

    # unpack arguments 
    path, data_type = args
    
    # construct save name 
    if (data_type == "stim"): 
        save_name = f"../../../../../../../../work3/s194329/results_real_{data_type}/ramp{path.split("/")[-1].split("e")[-1].split(".")[0]}.pkl"
    elif data_type == "ttx": 
        save_name = f"../../../../../../../../work3/s194329/results_real_{data_type}/eye{path.split(" ")[1][0]}.pkl"
    elif data_type == "spon": 
        save_name = f"../../../../../../../../work3/s194329/results_real_{data_type}/eye{path.split("/")[5][-1]}_ramp{path.split("/")[-1].split(".")[0].split("e")[1]}.pkl"

    
    if os.path.exists(save_name): 
        print(f"Skipping save_name : {save_name}")
        return

    # read data
    _, data = read_ns5_file(path)

    # determine if data should be binned 
    to_bin = False if data_type == "spon" else True

    # filter signal 
    print("filter signal")
    filtered_signal, idx = filter(data, bin = to_bin)

    # count CAPS using different methods 
    print("baseline and wavelet")
    estimated_caps_baseline = count_caps_baseline(np.delete(data, idx, axis = 1), filtered_signal, bin = to_bin)
    estimated_caps_wavelet = count_caps_wavelet(np.delete(data, idx, axis = 1), filtered_signal, bin = to_bin)

    # make new instance of simulator for SVM to train 
    print("svm")
    seed = hash(save_name) % (2**32)
    if bin: 
        simulator_train = SimulateData(1, [200, 5, 10, 20], CAP_dist="uniform", seed = seed, num_channels = 32*3)
    else: 
        simulator_train = SimulateData(1, [45, 0, 5, 10], CAP_dist=None, seed = seed, num_channels = 32*3)
    simulator_train.construct_signal()
    filtered_signal_train, _ = filter(simulator_train.signal, max_count=30, bin = to_bin)
    estimated_caps_svm = count_caps_svm(simulator_train, filtered_signal_train, filtered_signal, bin = to_bin)

    # get mean for mean predictor
    true_caps_train = count_true_caps(simulator_train)
    mean_predict = np.mean(np.sum(true_caps_train, axis = 0)) * np.ones(simulator_train.num_channels)

    # save to dictionary 
    d = {}
    d['estimated_baseline'] = estimated_caps_baseline 
    d['estimated_wavelet'] = estimated_caps_wavelet 
    d['estimated_svm'] = estimated_caps_svm  
    d['estimated_mean_predict'] = mean_predict


    # save files 
    print(f"saving {save_name}")
    with open(save_name, 'wb') as output_file: 
        pickle.dump(d, output_file)
    output_file.close()
    
def count_all(paths : list, data_type : str) -> None:
    inputs = [(p, data_type) for p in paths if p is not None]

    # start processing all files 
    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(counter, inputs) 
    
    # pool = Pool(processes=2)
    # pool.map(counter, inputs)


def flatten(l: list) -> list:
    return [subsublist for sublist in l for subsublist in sublist]

if __name__ == "__main__":
    # specify data type 
    data_type = "stim" # spon, ttx or stim

    # create folders to save results to 
    create_folders(data_type)

    # change directory
    os.chdir("src/data")


    # create paths 
    if data_type == "spon": 
        paths = flatten([[f"../../data/raw/Ramp data/Eye {i}/ramp_file{j}.ns5" for i in range(1, 7)] for j in [1, 12]])
    elif data_type == "stim": 
        paths = [f"../../data/raw/Ramp data/Eye 4/ramp_file{i}.ns5" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
    else:
        path = glob.glob("../../data/raw/3D/**/*.ns5")
        paths = [path[i] for i in range(len(path)) if "TTX" in path[i]]
    
    count_all(paths, data_type)

