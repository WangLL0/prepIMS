# prepIMS Example Usage

This document demonstrates a full example of using `prepIMS` on a test dataset. It is intended to help users quickly understand how to use each module in the pipeline and inspect the intermediate results.

## 📁 Example Dataset Structure

Place the following files in your working directory:

```
example_dataset/
├── Brain_DAN_TIMS_Neg_100um1.d/              # Raw Bruker data folder (optional if using extracted CSVs)
├── Brain_DAN_TIMS_Neg_100um1_poslog.txt/            # Spatial coordinate file (frame mapping)
├── 1/                 # Extracted CSVs per frame (output of extraction.py)
```

## 🔧 Step-by-Step Execution

### 1. Extract Raw IM-MSI Data

```bash
python extraction.py
```

This will parse the `.d` file and output individual `.csv` files per frame into the `frames/` folder.

### 2. Perform Peak Detection (Mass Domain)

```bash
python detection.py
```

This module scans each pixel's mass spectrum and detects candidate peaks.

### 3. Align Peaks Across All Pixels

```bash
python alignment.py
```

Aligns the detected peaks to form consistent feature sets across the sample.

### 4. Filter Low-Frequency Features

```bash
python filtering.py
```

Removes features that appear in only a small portion of the sample.

### 5. Build Sparse Matrix for Imaging

```bash
python sparsing.py
```

Constructs a sparse matrix (pixels × features) for visualization and analysis.

### 6. Visualize Ion Images

```bash
python imaging.py --selected_cluster 0
```

Generates and saves ion images for the selected feature cluster (e.g., cluster 0).

---

## ✅ Output

After running the full pipeline, you will find:

- `filtered_peaks.csv`: Final peak list after alignment and filtering
- `feature_matrix.npz`: Sparse matrix storing intensity values
- `ion_images/`: Folder containing PNG images of selected features

---

## 💡 Tips

- You can modify `selected_cluster` in `main.py` or `imaging.py` to visualize different feature groups.
- For a large dataset, consider batch processing or parallelization (not included by default).

---

## 📫 Contact

Questions? Contact **linlin_wang_z@163.com**