from src.model.wavelet_model import count_caps_wavelet
import pickle 
import glob 

def run_wavelet(files, save_path): 
    
    for file in files: 
        with open(file, 'rb') as f:
            data = pickle.load(f)

        all_est_counts = count_caps_wavelet(data, data)

        d = {}
        d['wavelet_res'] = all_est_counts

        save_name = save_path + file.split("/")[-1]

        print(f"saving {save_name}")
        with open(save_name, 'wb') as output_file: 
            pickle.dump(d, output_file)
        output_file.close()

if __name__ == "__main__":
    data_type = "stim"
    path = f"../../../../../../../../work3/s194329/results_synthetic_{data_type}/filtered_data/"
    save_path = f"../../../../../../../../work3/s194329/results_synthetic_{data_type}/wavelet_res/"

    files = sorted(glob.glob(path + "*.pkl"), key = len)
    run_wavelet(files, save_path)