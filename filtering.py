import numpy as np

def peak_pooling(data, thr = 0.05, save_txt=False):
    """
        Perform peak pooling to generate a feature matrix.

        Parameters:
            data (numpy.ndarray): The aligned peak data (columns: m/z, CCS, Intensity, Sample).
            thr (float, optional): The minimum occurrence ratio threshold for filtering peaks. Default is 0.05.
            save_txt (bool, optional): If True, save results to text files. Default is False.
        Returns:
            tuple: (Feature matrix, Filtered peak list)
    """
    sample = data[:, 3]
    intensity = data[:, 2]
    combine = data[:, 0:2]
    peak_list, count = np.unique(combine, axis=0, return_counts=True)
    data_len = int(np.max(sample) + 1)
    ratio = count / data_len
    peak_filter = peak_list[np.where(ratio > thr)]
    n = len(peak_filter)
    return_matrix = np.zeros((data_len, n))
    unique_sample = np.unique(sample).astype(int)
    for i in unique_sample:
        index = np.where(sample == i)
        for j in range(n):
            peak_index = np.where((combine[index] == peak_filter[j]).all(axis=1))[0]
            if len(peak_index) != 0:
                return_matrix[int(i), j] = np.max(intensity[index][peak_index]).astype(int)
    if save_txt:
        np.savetxt('FeatureMatrix', return_matrix, '%d')
        np.savetxt('2DPeaks_filtered', peak_filter, '%.4f %.3f')
    
    return return_matrix, peak_filter

if __name__ == '__main__':
    """
        If executed directly, load '2DPeaks_aligned.txt' and perform peak pooling.
    """
    data = np.loadtxt('2DPeaks_aligned')
    return_matrix, peak_filter = peak_pooling(data, save_txt=True)
