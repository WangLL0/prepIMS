import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os
import pathlib


def get_max_frame(poslog_path):
    """
        Reads the poslog.txt file and determines the max_frame count.

        Parameters:
            poslog_path (str): Path to the _poslog.txt file.

        Returns:
            int: (Number of unique (x, y) + 1) coordinate pairs (max_frame).
        """
    r = []
    with open(poslog_path, 'r') as file:
        file.readline()
        for line in file:
            s = line.split(' ')
            if len(s) > 2 and s[2] != '__':
                try:
                    a = s[2].split('X')[1]
                    x = int(a.split('Y')[0])
                    y = int(a.split('Y')[1])
                    if (x, y) not in r:
                        r.append((x, y))
                except (IndexError, ValueError):
                    continue

    return len(r) + 1

def get_distance_matrix(datas):
    n = len(datas)
    dists = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.abs(datas[i] - datas[j])
    return dists

def select_dc(dists, dc_percent=10):
    n = np.shape(dists)[0]
    tt = np.reshape(dists, n * n)
    position = int(n * n * dc_percent / 100 + 1)
    dc = np.sort(tt)[position + n]
    return dc

def get_density(dists, dc, method=None):
    n = np.shape(dists)[0]
    rho = np.zeros(n)
    for i in range(n):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


def get_deltas(dists, intensity):
    n = np.shape(dists)[0]
    deltas = np.zeros(n)
    for i, index in enumerate(intensity):
        if i < 30:
            neighbors_intensity_left = intensity[:i]
            neighbors_intensity_right = intensity[i + 1: i + 61 - len(neighbors_intensity_left)]

        elif i >= len(intensity) - 30:
            neighbors_intensity_right = intensity[i + 1:]
            neighbors_intensity_left = intensity[i+len(neighbors_intensity_right)-60:i]

        else:
            neighbors_intensity_right = intensity[i-30:i]
            neighbors_intensity_left = intensity[i+1:i+31]

        neighbors_intensity = np.concatenate((neighbors_intensity_left, neighbors_intensity_right))
        del_intensity = intensity[i] - neighbors_intensity

        if np.any(del_intensity < 0):
            deltas[i] = 0
        else:
            average_intensity = np.mean(intensity[i] - neighbors_intensity)
            deltas[i] = average_intensity

    return deltas

def find_centers_auto(rho, delta, ccs, intensity, density_percent=5):

    sorted_indices = np.argsort(intensity)
    sorted_rho = rho[sorted_indices]
    sorted_delta = delta[sorted_indices]
    position = int(len(ccs) * density_percent / 100 + 1)
    rho_threshold = np.mean(sorted_rho[:position])
    delta_threshold = np.mean(sorted_delta[:position])

    n = np.shape(rho)[0]
    centers = []
    ccs_data = []
    intensity_data = []

    for i in range(n):
        if rho[i] >= rho_threshold and delta[i] > delta_threshold:
            centers.append(i)
            ccs_data.append(ccs[i])
            intensity_data.append(intensity[i])

    intensity_data = np.array(intensity_data)
    sorted_indices = np.argsort(-intensity_data)
    sorted_centers = np.array(centers)[sorted_indices]
    sorted_ccs_data = np.array(ccs_data)[sorted_indices]
    sorted_intensity_data = np.array(intensity_data)[sorted_indices]

    m = 0
    for i in range(1, len(sorted_indices)):
        current_ccs = sorted_ccs_data[i-m]
        inside_previous_boundaries = False
        for j in range(i-m):
            left_ccs = sorted_ccs_data[j] - 0.005*sorted_ccs_data[j]
            right_ccs = sorted_ccs_data[j] + 0.005*sorted_ccs_data[j]

            if left_ccs <= current_ccs <= right_ccs:
                sorted_ccs_data = np.delete(sorted_ccs_data, i - m)
                sorted_centers = np.delete(sorted_centers, i - m)
                sorted_intensity_data = np.delete(sorted_intensity_data, i - m)
                m += 1
                break
    return np.array(sorted_centers)

def draw_decision(rho, deltas, datas, mzpeaks, dc):
    plt.figure(figsize=(8, 4))
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.title('mz:' + str(mzpeaks) + '        dc:%s' % str(dc))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.show()

def combining_data(ccs, intensity):

    return_ccs, indices = np.unique(ccs, return_inverse=True)
    return_intensity = np.bincount(indices, weights=intensity)
    return np.array(return_ccs), np.array(return_intensity)

def mz_peaks(mz, intensity, snr=5, ppm_threshold=40):
    """
        Identifies m/z peaks from intensity data using a signal-to-noise ratio (SNR) threshold.

        Parameters:
            mz (numpy.ndarray): Array of m/z values.
            intensity (numpy.ndarray): Corresponding intensity values.
            snr (float, optional): Signal-to-noise ratio threshold for peak selection (default = 5).
            ppm_threshold (float, optional): Maximum allowed difference between peaks in ppm (default = 40 ppm).

        Returns:
            numpy.ndarray: Selected m/z peak values.
    """
    mz_combine, mz_intensity_combine = combining_data(mz, intensity)
    mz_array = mz_combine[np.argsort(mz_combine)]
    mz_intensity_array = mz_intensity_combine[np.argsort(mz_combine)]
    median_value = np.median(mz_intensity_array)
    abs_deviation = np.abs(mz_intensity_array - median_value)
    mad = np.median(abs_deviation)
    n = 1.4826 * mad
    s = snr * n
    peaks, _ = find_peaks(mz_intensity_array)
    peak_x = mz_array[peaks]
    peak_y = mz_intensity_array[peaks]

    flag = 0
    while flag == 0:
        try:
            indices_to_remove = np.where(np.abs((peak_x[1:] - peak_x[:-1]) / (
                    (peak_x[1:] + peak_x[:-1]) / 2)) < ppm_threshold)[0][0]
        except:
            indices_to_remove = -1
        if indices_to_remove >= 0:
            if peak_y[indices_to_remove] < peak_y[indices_to_remove + 1]:
                peak_x = np.delete(peak_x, indices_to_remove)
                peak_y = np.delete(peak_y, indices_to_remove)
            else:
                peak_x = np.delete(peak_x, indices_to_remove + 1)
                peak_y = np.delete(peak_y, indices_to_remove + 1)
        else:
            flag = 1
    mzpeaks_final = peak_x[np.where(peak_y > s)]
    intensitypeaks_final = peak_y[np.where(peak_y > s)]

    return mzpeaks_final[np.argsort(-intensitypeaks_final)]



def ccs_peaks(mz, ccs, intensity, mzpeaks, tol_mz=20, tol_ccs = 0.005):
    """
        Identifies CCS peaks by clustering them based on density and intensity.

        Parameters:
            mz (numpy.ndarray): Array of m/z values.
            ccs (numpy.ndarray): Array of CCS values.
            intensity (numpy.ndarray): Corresponding intensity values.
            mzpeaks (numpy.ndarray): Selected m/z peaks from mz_peaks function.
            tol_mz (int, optional): m/z tolerance in ppm for CCS clustering (default = 20 ppm).
            ppm_threshold (float, optional): Maximum allowed difference between peaks in ppm (default = 40 ppm).
            tol_ccs (float, optional): Tolerance for CCS matching (default=0.005).

        Returns:
            tuple: (Selected m/z peaks, Corresponding CCS values, Intensity values).
    """
    return_mz = []
    return_ccs = []
    return_intensity = []
    for i in range(len(mzpeaks)):
        peaks = np.where((mz >= mzpeaks[i] * (1000000 - tol_mz) / 1000000) & (mz <= mzpeaks[i] * (1000000 + tol_mz) / 1000000))
        ccs_select = ccs[peaks]
        intensity_select = intensity[peaks]
        ccs_combine, ccs_intensity_combine = combining_data(ccs_select, intensity_select)

        if len(ccs_combine) == 1:
            return_mz.append(mzpeaks[i])
            return_ccs.append(0)
            return_intensity.extend(ccs_intensity_combine)

        else:
            dists = get_distance_matrix(ccs_combine)
            dc = select_dc(dists, dc_percent=10)
            rho = get_density(dists, dc, method="Gaussion")
            delta = get_deltas(dists, ccs_intensity_combine)
            centers = find_centers_auto(rho, delta, ccs_combine, ccs_intensity_combine, density_percent=5)

            if len(centers) > 0:
                intensity_peak = []
                for j in range(len(centers)):
                    index = centers[j]
                    intensity_peak.append(np.sum(ccs_intensity_combine[np.where((ccs_combine >= ccs_combine[index] * (1 - tol_ccs)) & (
                                                                            ccs_combine <= ccs_combine[index] * (1 + tol_ccs)))]))
                intensity_peak = np.array(intensity_peak)
                return_ccs.extend(ccs_combine[centers])
                return_intensity.extend(intensity_peak)
                return_mz.extend([mzpeaks[i] for m in range(len(centers))])
            else:
                return_mz.append(mzpeaks[i])
                return_ccs.append(0)
                return_intensity.append(np.sum(ccs_intensity_combine))

    return np.array(return_mz), np.array(return_ccs), np.array(return_intensity)

def peakpicking(data, max_frame):
    """
        Perform peak picking on IM-MSI data.

        Parameters:
            data (numpy.ndarray): The full dataset as a NumPy array.
            max_frame (int): The maximum frame number.

        Returns:
            numpy.ndarray: An array containing peak results with frame indices.
    """
    all_results = []  # Store peak results

    for i in range(1, max_frame + 1):
        frame_data = data[data[:, 0] == i]  # Extract data for this frame

        if frame_data.size > 0:
            peak_results = pipeline(i, frame_data, use_rawdata=False, save_file=False)

            if peak_results.size > 0:
                frame_indices = np.full((peak_results.shape[0], 1), i, dtype=int)
                frame_results = np.hstack((frame_indices, peak_results))
                all_results.extend(frame_results.tolist())

    final_results = np.array(all_results) if all_results else np.empty((0, 4), dtype=np.float64)
    return final_results



def pipeline(i, frame_data=None, use_rawdata=True, save_file=False):
    """
        Pipeline for peak picking, either from RawData CSV or in-memory data.

        Parameters:
            i (int): Frame index.
            frame_data (numpy.ndarray, optional): The data for this frame (used if `use_rawdata=False`).
            use_rawdata (bool): If True, load data from 'RawData' directory; otherwise, use in-memory data.

        Returns:
            numpy.ndarray: The peak picking results.
    """
    if use_rawdata:
        # Load from RawData CSV when executing directly
        data = pd.read_csv(f'RawData/{i}.csv', header=None).values
        print(f'Processing frame: {i}.csv from RawData')
    else:
        # Use provided in-memory data when called from main.py
        data = frame_data

    intensity = data[:, 1].astype(int)
    mz = np.around(data[:, 2].astype(float), decimals=6)
    ccs = np.around(data[:, 3].astype(float), decimals=6)

    mzpeaks = mz_peaks(mz, intensity, snr=5, ppm_threshold=40)
    mz_peak, ccs_peak, intensity_peak = ccs_peaks(mz, ccs, intensity, mzpeaks)
    mz_peak_rounded = np.around(mz_peak, decimals=6)
    ccs_peak_rounded = np.around(ccs_peak, decimals=6)
    output_data = np.column_stack((mz_peak_rounded, ccs_peak_rounded, intensity_peak))

    # **Only save CSV if script is executed directly (not when called from main.py)**
    if save_file:
        folder_path = '2DPeaks'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        output_file = f'{folder_path}/output{i}.csv'
        np.savetxt(output_file, output_data, delimiter=',', fmt=['%.6f', '%.6f', '%d'], comments='')

    return output_data




if __name__ == '__main__':
    """
        If the script is run directly, it will process frames from 'RawData' directory and save the results to '2DPeaks/'.
        If called from main.py, it will use in-memory data and NOT save CSVs.
    """
    file_path = "/path/to/your/data.d"
    poslog_path = pathlib.Path(file_path).with_suffix('')  # Remove .d suffix
    poslog_path = str(poslog_path) + "_poslog.txt"
    max_frame = get_max_frame(poslog_path)

    for i in range(1, max_frame):
        pipeline(i, use_rawdata=True, save_file=True)  # Read data from RawData CSV files and save results


