# prepIMS

prepIMS is a preprocessing workflow designed for IM-MSI (Ion Mobility-Mass Spectrometry Imaging) data. It integrates key processing steps such as data extraction, peak detection, alignment, feature filtering, and spatial mapping, ensuring efficient and accurate downstream analysis. The workflow is highly modular, allowing seamless adaptation to different IM-MSI datasets and applications.
Developer is Linlin Wang from Xiamen University of China.

# Overflow of prepIMS Model

<div align=center>
<img src="https://raw.githubusercontent.com/WangLL0/prepIMS/refs/heads/main/read.png" width="650" height="650" /><br/>
</div>

__Workflow of the prepIMS for 4D MSI Data Preprocessing__. (A) Acquisition of IM-MSI data from whole-body mouse pup tissue; (B) Extraction of mass and ion mobility signals for each pixel; (C) Peak detection for the mass signals; (D) Extraction of the ion mobility signals from the specific mass peak within 20 ppm; (E) Calculation of the density index (rho) and relative intensity index (delta) for each ion mobility data point; (F) Peak detection for ion mobility signals based on calculated rho and delta values; (G) Grouping and merging 2D signals from all pixels to obtain the intensity of each peak; (H) Performing 2D peak alignment and filtering to reconstruct the ion images.  

# Requirement

    python == 3.9

    conda install numpy=1.26.4 scipy=1.13.1 pandas=2.2.2 scikit-learn=1.6.1

    conda install matplotlib=3.8.4 seaborn

    conda install umap-learn=0.5.7 -c conda-forge

    conda install scipy=1.13.1

    pip install timspy==0.9.3

    pip install opentims-bruker-bridge opentimspy

    
# Quickly start

## Input
The input consists of raw IM-MSI data stored in a .d file, which contains full spectral and ion mobility information. Additionally, a poslog.txt file is provided, containing spatial coordinate mapping for each frame.

## Run prepIMS model

cd to the prepIMS fold

If you want to run the entire end-to-end workflow to obtain the final processed results and visualize the highly correlated features of the N-th cluster, run:

    python main.py --selected_cluster N

If you want to parse .d data and save the results of each frame as a .csv file, run:

    python extraction.py

If you want to obtain the peak detection results for each pixel, run:

    python detection.py

If you want to align the peak detection results, run:

    python alignment.py

If you want to further filter peaks based on their frequency, run:

    python filtering.py

If you want to further filter peaks based on their frequency, run:

    python filtering.py

If you want to generate a sparse matrix for molecular imaging, run:

    python sparsing.py

If you want to visualize molecular feature images for each selected feature, run:

    python imaging.py --selected_cluster N


# Contact

Please contact me if you have any help: linlin_wang_z@163.com
