import scipy.io as scio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
import pywt
import random
import shutil


def move_random_files(src_folder, dest_folder, ratio=0.2):
    if not os.path.exists(src_folder):
        raise ValueError(f"源文件夹不存在: {src_folder}")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)

    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    if not all_files:
        print(f"源文件夹 {src_folder} 中没有文件可移动。")
        return
    num_files_to_move = int(len(all_files) * ratio)
    files_to_move = random.sample(all_files, num_files_to_move)

    for file_name in files_to_move:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.move(src_path, dest_path)

    print(f"\n总计移动了 {len(files_to_move)} 个文件到 {dest_folder}。")


def signal_to_stft_image(signal, output_path, nperseg=256, noverlap=128, cmap='jet'):
    f, t, Zxx = stft(signal, window='hann', nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cmap)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def signal_to_cwt_image(signal, output_path, wavelet='cmor1.5-1.0', scales=None, cmap='jet'):
    fs = 64000
    if scales is None:
        scales = np.arange(1, 128)

    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.pcolormesh(np.linspace(0, len(signal) / fs, len(signal)), frequencies, np.abs(coefficients),
                   shading='gouraud', cmap=cmap)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_phase_current():
    def load_mat_return_array(path, time_step=512, seq_len=1024):
        keys = []
        data = scio.loadmat(path)
        for key in data.keys():
            keys.append(key)
        X = data[keys[3]]
        idx = []
        for i, _ in enumerate(X):
            if i % time_step == 0:
                idx.append(i)
        idx = idx[:-2]
        data = []
        for id in idx:
            if len(X[id: id + seq_len]) < seq_len:
                continue
            else:
                data.append(X[id: id + seq_len])
        return np.array(data)

    from tqdm import tqdm

    namelist = os.listdir("CWRU")
    for _, name, label in zip(tqdm(range(10)), namelist, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        mat_path = os.path.join("CWRU", name)
        name = name.split(".")[1]
        if name != "mat":
            name1, name2 = name.split("-")[0], name.split("-")[1]
            name = f"{name2}_{name1}"
        else:
            name = "normal"
        data_only = load_mat_return_array(mat_path)
        for u, one_seq in enumerate(data_only):
            one_seq = one_seq.reshape(-1, )
            signal_to_stft_image(signal=one_seq, output_path=f"../stft/train/{name}_{u}_{label}")
            signal_to_cwt_image(signal=one_seq, output_path=f"../cwt/train/{name}_{u}_{label}")


if __name__ == '__main__':
    process_phase_current()
