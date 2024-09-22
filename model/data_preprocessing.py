#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, get_window

# Normalization under Different Constant Speed
def normalize_signal(v, f_r):

    v_normalized = v / (f_r ** 2)
    mu = np.mean(v_normalized)
    sigma = np.std(v_normalized)
    v_n = (v_normalized - mu) / sigma
    return v_n

# Resampling after Normalization (using Cubic Spline Interpolation)
def resample_signal(theta, v_n, new_theta):

    cs = CubicSpline(theta, v_n)
    v_resampled = cs(new_theta)
    return v_resampled

# Envelope calculation using Hilbert Transform
def envelope_signal(v):

    analytic_signal = hilbert(v)
    envelope = np.abs(analytic_signal)
    return envelope

# Calculate window length W based on f_s and frequency peak separation Δf_peaks
def calculate_window_length(f_s, f_max, f_peaks):

    # Use Nyquist sampling frequency
    f_s_min = 2 * f_max
    # Ensure sampling frequency is at least Nyquist
    if f_s < f_s_min:
        f_s = f_s_min
    # Calculate window length based on formula W ≥ (2 * f_max) / Δf_peaks
    W = int((2 * f_max) / f_peaks)
    return W

# Apply window function
def apply_window_function(v, W):

    window = get_window('hann', W)  
    v_windowed = v[:W] * window 
    return v_windowed

# Combine everything into a complete preprocessing function for network input
def preprocess_signal(v, f_r, f_s, f_max, f_peaks):

    # Step 1: Normalize the signal
    v_n = normalize_signal(v, f_r)

    # Step 2: Resample the signal
    theta = np.linspace(0, 2 * np.pi, len(v))  # Assume linear theta for simplicity
    new_theta = np.linspace(0, 2 * np.pi, len(v) * 2)  # More resampling points for precision
    v_resampled = resample_signal(theta, v_n, new_theta)

    # Step 3: Calculate the envelope of the resampled signal
    v_envelope = envelope_signal(v_resampled)

    # Step 4: Calculate window length W
    W = calculate_window_length(f_s, f_max, f_peaks)
    
    # Step 5: Apply window function
    v_windowed = apply_window_function(v_envelope, W)
    
    return v_windowed

# Example usage
if __name__ == "__main__":
    # Simulate some sample data
    t = np.linspace(0, 1, 1000)  # Time vector
    f_r = 10  # Constant rotational frequency (example value)
    f_s = 1000  # Sampling frequency (example)
    f_max = 100  # Maximum frequency in the signal (example)
    f_peaks = 5  # Frequency peak separation (example)
    
    # Simulated vibration signal
    v = np.sin(2 * np.pi * f_r * t) + 0.1 * np.random.normal(size=t.shape)

    # Preprocess the signal
    v_preprocessed = preprocess_signal(v, f_r, f_s, f_max, f_peaks)

    # Print the result (or plot it using matplotlib for visualization)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(v_preprocessed, label='Preprocessed Signal')
    plt.title('Preprocessed Vibration Signal with Windowing')
    plt.legend()
    plt.show()

