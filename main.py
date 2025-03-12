import pathlib
import extraction  # Module for extracting IM-MSI data
import detection   # Module for peak detection
import alignment   # Module for aligning detected peaks
import filtering   # Module for filtering significant features
import sparsing    # Module for generating sparse matrix for spatial mapping
import imaging     # Module for feature selection and imaging


if __name__ == "__main__":
    """
        Main workflow for IM-MSI data processing. This script follows these steps:
        1. Extract raw data from .d files
        2. Perform peak picking to identify significant mass/charge features
        3. Align detected peaks across all frames
        4. Filter features based on their occurrence across frames
        5. Generate a sparse matrix for spatial mapping
        6. Perform feature selection and imaging
    """

    file_path = '/path/to/your/data.d'  # Change this to your actual .d file path
    poslog_path = pathlib.Path(file_path).with_suffix('')
    poslog_path = str(poslog_path) + "_poslog.txt"
    selected_clusters = 3
    data_array, max_frame = extraction.process_data(file_path)  # Process IM-MSI data
    print("\nRunning Peak Picking on Extracted Data...")
    peak_results = detection.peakpicking(data_array, max_frame)
    print("\nRunning Alignment on Peak Results...")
    aligned_results = alignment.alignment_pipeline(peaks=peak_results)
    print("\nRunning Feature Filtering on Aligned Data...")
    feature_matrix, filtered_peaks = filtering.peak_pooling(data=aligned_results)
    print("\nGenerating Sparse Matrix for Spatial Mapping...")
    x_grid, y_grid, sparse_matrix = sparsing.generate_sparse_matrix(feature_matrix, poslog_path)
    print("\nRunning Feature Selection & Imaging...")
    auc_scores, peak, sorted_indices = imaging.feature_selection(m=x_grid, n=y_grid, selected_cluster=selected_clusters,
                                                                 data=sparse_matrix, peak=filtered_peaks)
