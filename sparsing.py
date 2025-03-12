import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import pathlib

def generate_sparse_matrix(data, poslog_file, save_file=False):
    """
        Generate a sparse matrix from IM-MSI data and positional log.

        Parameters:
            data (numpy.ndarray): The feature matrix to be converted (typically the aligned `FeatureMatrix`).
            poslog_file (str): Path to the position log file (e.g., 'Baby_dan_tims_pos_50um_poslog.txt').
            save_file (bool): Whether to save the sparse matrix as a `.npz` file (default: False).

        Returns:
            tuple: (X-axis grid size, Y-axis grid size, generated sparse matrix `csr_matrix`)
    """
    # Read position log file
    c = open(poslog_file, 'r')
    c.readline()  # 丢弃第一行
    line = c.readline()
    r = []
    X = []
    Y = []
    while line != '':
        s = line.split(' ')
        if s[2] != '__':
            a = s[2].split('X')[1]
            x = a.split('Y')[0]
            y = a.split('Y')[1]
            x = int(x)
            y = int(y)

            if len(X) > 0:
                if not (int(x) == X[(len(X) - 1)] and int(y) == Y[(len(Y) - 1)]):
                    X.append(int(x))
                    Y.append(int(y))
                    r.append((x, y))
            else:
                X.append(int(x))
                Y.append(int(y))
                r.append((x, y))
        line = c.readline()

    X, Y = np.array(X), np.array(Y)
    c = np.array(r)
    xy_min = np.min(c, axis=0)
    cc = c - xy_min


    # Create sparse matrix
    num_rows = (np.max(cc[:, 0]) + 1) * (np.max(cc[:, 1]) + 1)
    num_columns = data.shape[1]
    new_matrix = np.zeros((num_rows, num_columns))

    for i in range(np.max(cc[:, 0]) + 1):
        for j in range(np.max(cc[:, 1]) + 1):
            indices = np.where((cc[:, 0] == i) & (cc[:, 1] == j))[0]
            if indices.size > 0:
                index = indices[0]
                new_matrix[i * (np.max(cc[:, 1]) + 1) + j, :] = data[index, :]
            else:
                new_matrix[i * (np.max(cc[:, 1]) + 1) + j, :] = np.zeros(num_columns)

    sparse_matrix = csr_matrix(new_matrix)
    if save_file:
        sparse.save_npz('sparse_matrix.npz', sparse_matrix)

    return (np.max(cc[:, 0]) + 1), (np.max(cc[:, 1]) + 1), sparse_matrix

if __name__ == "__main__":
    file_path = "/path/to/your/data.d"
    poslog_path = pathlib.Path(file_path).with_suffix('')  # Remove .d suffix
    poslog_path = str(poslog_path) + "_poslog.txt"
    data_path = 'FeatureMatrix'
    data = np.loadtxt(data_path)
    generate_sparse_matrix(data, poslog_path, save_file=True)