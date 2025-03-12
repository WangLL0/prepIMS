import numpy as np
import pandas as pd
import os


def binning_data(data_mz, data_intensity, tolerance=20, d=6):
    """
        Perform binning (alignment) on m/z or CCS data.

        Parameters:
            data_mz (numpy.ndarray): m/z or CCS values.
            data_intensity (numpy.ndarray): Corresponding intensity values.
            tolerance (int or float): The tolerance for merging values (in ppm for m/z, absolute for CCS).
            d (int): Number of decimal places for rounding.

        Returns:
            numpy.ndarray: Binned (aligned) m/z or CCS values.
    """
    mz_unique, indices = np.unique(data_mz, return_inverse=True)
    mz_unique_intensity = np.bincount(indices, weights=data_intensity)
    while len(mz_unique) > 0:
        max_intensity_index = np.argmax(mz_unique_intensity)
        max_intensity_mz = mz_unique[max_intensity_index]
        if tolerance >= 1:
            indices_within_tolerance_datamz = np.where((data_mz >= max_intensity_mz * (1000000 - tolerance) / 1000000) & (data_mz <= max_intensity_mz * (1000000 + tolerance) / 1000000))[0]
            indices_within_tolerance_mzunique = np.where((mz_unique >= max_intensity_mz * (1000000 - tolerance) / 1000000) & (mz_unique <= max_intensity_mz * (1000000 + tolerance) / 1000000))[0]
        else:
            indices_within_tolerance_datamz = np.where((data_mz >= max_intensity_mz * (1 - tolerance)) & (data_mz <= max_intensity_mz * (1 + tolerance)))[0]
            indices_within_tolerance_mzunique = np.where((mz_unique >= max_intensity_mz * (1 - tolerance)) & (mz_unique <= max_intensity_mz * (1000000 + tolerance)))[0]
        mean_mz = np.around(np.mean(data_mz[indices_within_tolerance_datamz]), d)
        data_mz[indices_within_tolerance_datamz] = mean_mz
        mz_unique = np.delete(mz_unique, indices_within_tolerance_mzunique)
        mz_unique_intensity = np.delete(mz_unique_intensity, indices_within_tolerance_mzunique)
    return np.array(data_mz)

def alignment_pipeline(use_rawdata=False, peaks=None, save_txt=False):
    if use_rawdata:
        folder_path = '2DPeaks'
        file_names = os.listdir(folder_path)
        data_total = np.zeros((1,3))
        samples = []
        num = 0
        # 遍历文件夹中的文件，并读取数据
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):  # 确保是文件而不是文件夹
                data = pd.read_csv(file_path, header=None).values  # 读取文件数据
                mz = data[0:, 0]
                ccs = data[0:, 1]
                intensity = data[0:, 2]
                intensity = np.array([int(m) for m in intensity])
                mz = np.array([float(m) for m in mz])
                ccs = np.array([float(m) for m in ccs])
                data = np.vstack((mz, ccs, intensity)).T
                data_total = np.concatenate((data_total, data), axis=0)
                file = int(file_name[6:][:-4])
                samples.extend([file] * len(data))
        data_total = data_total[1:]
        data_mz = data_total[:, 0]
        data_ccs = data_total[:, 1]
        data_intensity = data_total[:, 2]
        data_sample = np.array(samples)

    else:
        data = peaks
        samples = data[0:, 0]
        mz = data[0:, 1]
        ccs = data[0:, 2]
        intensity = data[0:, 3]
        data_intensity = np.array([int(m) for m in intensity])
        data_mz = np.array([float(m) for m in mz])
        data_ccs = np.array([float(m) for m in ccs])
        data_sample = np.array([int(m) for m in samples])
    print(data_mz)
    print(data_ccs)
    print(data_intensity)
    print(data_sample)

    data_mz = binning_data(data_mz, data_intensity, tolerance=20, d=6)
    data_mz_unique, indices = np.unique(data_mz, return_inverse=True)
    for i in range(len(data_mz_unique)):
        ccs_index = np.where(data_mz == data_mz_unique[i])[0]
        data_mz_ccs = data_ccs[ccs_index]
        data_mz_ccs_intensity = data_intensity[ccs_index]
        data_mz_ccs = binning_data(data_mz_ccs, data_mz_ccs_intensity, tolerance=0.005, d=6)
        data_ccs[ccs_index] = data_mz_ccs
    sorted_indices = np.argsort(data_sample).astype(int)
    data_mz_sorted = data_mz[sorted_indices]
    data_ccs_sorted = data_ccs[sorted_indices]
    data_intensity_sorted = data_intensity[sorted_indices]
    data_sample_sorted = data_sample[sorted_indices]
    data_aligned = np.hstack((np.array(data_mz_sorted).reshape(-1, 1), np.array(data_ccs_sorted).reshape(-1, 1),
                      np.array(data_intensity_sorted).reshape(-1, 1), np.array(data_sample_sorted).reshape(-1, 1)))
    if save_txt:
        np.savetxt('2DPeaks_aligned', data_aligned, fmt='%.6f %.6f %d %d')
    return data_aligned

if __name__ == "__main__":
    alignment_pipeline(use_rawdata=True, peaks=None, save_txt=True)