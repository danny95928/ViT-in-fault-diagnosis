import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
import pywt


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# 模拟信号数据
# 示例：生成一个包含两个不同频率的正弦波信号
fs = 1000  # 采样频率 (Hz)
t = np.linspace(0, 2, fs * 2, endpoint=False)  # 时间轴
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # 信号 = 50Hz + 120Hz

# STFT 计算
f, t_stft, Zxx = stft(signal, fs=fs, window='hann', nperseg=256, noverlap=128)

# 绘制 STFT 结果（时频图）
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='jet')
plt.title('STFT - 时频图')
plt.ylabel('频率 (Hz)')
plt.xlabel('时间 (s)')
plt.colorbar(label='幅值')
plt.tight_layout()
plt.savefig("stft_image.png")  # 保存为图像文件
plt.show()


# 模拟信号数据
# 示例：生成一个线性调频信号（chirp信号）
t = np.linspace(0, 1, 1000, endpoint=False)  # 时间轴
signal = chirp(t, f0=10, f1=100, t1=1, method='linear')  # 线性调频信号

# CWT 计算
scales = np.arange(1, 128)  # 小波变换的尺度
coefficients, frequencies = pywt.cwt(signal, scales, 'cmor', sampling_period=1/1000)

# 绘制 CWT 结果（时频图）
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, frequencies, np.abs(coefficients), shading='gouraud', cmap='jet')
plt.title('CWT - 时频图')
plt.ylabel('频率 (Hz)')
plt.xlabel('时间 (s)')
plt.colorbar(label='幅值')
plt.tight_layout()
plt.savefig("cwt_image.png")  # 保存为图像文件
plt.show()



