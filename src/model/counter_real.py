from src.model.baseline_model import count_caps_baseline
from src.model.wavelet_model import count_caps_wavelet
from src.model.svm_model import count_caps_svm
from src.data.create_simulated_data import SimulateData
from src.data.preprocess_utils import filter, read_ns5_file
from multiprocessing import Pool
import numpy as np 
import pickle 
import os 


def create_folders() -> None:
    prefix = "../../../../../../../../work3/s194329/"
    os.makedirs(prefix + "results", exist_ok=True)

def counter(args : tuple[str]) -> None: 

    # unpack arguments 
    path = args
    
    # construct save name 
    save_name = f"../../../../../../../../work3/s194329/results/ramp{path.split(".")[-2][-1]}.pkl"

    if os.path.exists(save_name): 
        print(f"Skipping save_name : {save_name}")
        return

    # read data
    _, data = read_ns5_file(path)

    # filter signal 
    print("filter signal")
    filtered_signal, idx = filter(data)

    # count CAPS using different methods 
    print("baseline and wavelet")
    estimated_caps_baseline = count_caps_baseline(np.delete(data, idx, axis = 1), filtered_signal)
    estimated_caps_wavelet = count_caps_wavelet(np.delete(data, idx, axis = 1), filtered_signal)

    # make new instance of simulator for SVM to train 
    print("svm")
    seed = hash(save_name) % (2**32)
    simulator_train = SimulateData(1, [200, 5, 10, 20], CAP_dist="uniform", seed = seed)
    simulator_train.construct_signal()
    filtered_signal_train, _ = filter(simulator_train.signal)
    estimated_caps_svm = count_caps_svm(simulator_train, filtered_signal_train, filtered_signal)

    # save to dictionary 
    d = {}
    d['estimated_baseline'] = estimated_caps_baseline 
    d['estimated_wavelet'] = estimated_caps_wavelet 
    d['estimated_svm'] = estimated_caps_svm  

    # save files 
    print(f"saving {save_name}")
    with open(save_name, 'wb') as output_file: 
        pickle.dump(d, output_file)
    output_file.close()
    
def count_all(paths):
    inputs = [(p) for p in paths if p is not None]

    # start processing all files 
    with Pool() as pool:
        # Process 2D and 3D data
        result = pool.map(counter, inputs) 
    
    # pool = Pool(processes=2)
    # pool.map(counter, inputs)

if __name__ == "__main__":
    # create folders to save results to 
    create_folders()

    # change directory
    os.chdir("src/data")

    # run the counting 
    paths = [f"../../data/raw/Ramp data/Eye 5/ramp_file{i}.ns5" for i in range(2, 12)]
    count_all(paths)

