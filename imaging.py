import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def feature_selection(m, n, selected_cluster, sparse_data, peak):
    """
    Perform UMAP-based feature selection and clustering on IM-MSI data.

    Parameters:
        m (int): Number of rows in the grid.
        n (int): Number of columns in the grid.
        selected_cluster (int): The cluster index to analyze.
        data (numpy.ndarray)
        peak (numpy.ndarray): Peak feature array.
    """
    data = sparse_data.toarray()
    umap_features = umap.UMAP(n_components=3, metric='cosine', random_state=0).fit_transform(data)
    umap_features = MinMaxScaler().fit_transform(umap_features)  # Normalize values

    plt.imshow(umap_features.reshape(m, n, 3))
    plt.axis('off')
    plt.title("UMAP Feature Representation")
    plt.show()

    kmeans_labels = KMeans(n_clusters=20, random_state=0).fit_predict(umap_features)

    trainlabel = np.where(kmeans_labels == selected_cluster, 1, 0)
    plt.imshow(trainlabel.reshape(m, n))
    plt.show()

    aucs = []
    for i in range(len(peak)):
        train = data[:, i]
        lr = LogisticRegression()
        lr.fit(train.reshape(-1, 1), trainlabel.reshape(-1, 1))
        yPredS = lr.predict_proba(train.reshape(-1, 1))
        aucs.append(roc_auc_score(trainlabel, yPredS[:, 1]))

    aucs = np.array(aucs)
    index = np.argsort(-aucs)
    peak = np.around(peak, 6)
    for i in range(len(index)):
        plt.imshow(data[:, index[i]].reshape(m, n))
        plt.axis('off')
        plt.title(peak[index[i]])
        plt.show()

if __name__ == "__main__":
    sparse_data = sparse.load_npz('sparse_matrix.npz')
    peak_data = np.loadtxt('2DPeaks_filtered')

    auc_scores, peak, sorted_indices = feature_selection(m=255, n=135, selected_cluster=12,  sparse_data=sparse_data, peak=peak_data)
