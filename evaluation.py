#import packages
import os
import scanpy as sc
import pandas as pd
import squidpy as sq
import numpy as np
from scipy.spatial import *
from sklearn.preprocessing import *

from sklearn.metrics import *
from scipy.spatial.distance import *


from SDMBench import sdmbench


# Define the directory containing the clustering result files and .h5ad files
clustering_results_dir = "D:/code/CMML/ICA2/benchmark/output/louvain"  # Replace with the correct directory
adata_files_dir = 'D:/code/CMML/ICA2/Data'  # Replace with the correct directory
output_file = './output_results_all_2.csv'  # This will store all results

# Create a list to store all results for each combination of .h5ad and CSV files
all_results = []

# Iterate through each .h5ad file in the directory
for adata_file in os.listdir(adata_files_dir):
    if adata_file.endswith(".h5ad"):  # Only process the relevant .h5ad files
        adata_file_path = os.path.join(adata_files_dir, adata_file)
        
        # Load the adata for this .h5ad file
        adata = sc.read_h5ad(adata_file_path)
        adata_valid = adata[np.logical_not(adata.obs['Region'].isna())]  # Remove rows with missing region data
        
        # Get the corresponding clustering result file
        clustering_file_name = adata_file.replace(".h5ad", "_clustering_results.csv")
        clustering_file_path = os.path.join(clustering_results_dir, clustering_file_name)
        
        # Check if the clustering result file exists
        if os.path.exists(clustering_file_path):
            # Load the predicted clustering results from the CSV
            pred = pd.read_csv(clustering_file_path)
            adata_valid.obs['pred'] = pred.values
            adata_valid.obs['pred'] = adata_valid.obs['pred'].astype('category')
            
            # Compute the metrics
            nmi = sdmbench.compute_NMI(adata_valid, 'Region', 'pred')
            hom = sdmbench.compute_HOM(adata_valid, 'Region', 'pred')
            com = sdmbench.compute_COM(adata_valid, 'Region', 'pred')
            chaos = sdmbench.compute_CHAOS(adata_valid, 'pred')
            pas = sdmbench.compute_PAS(adata_valid, 'pred', spatial_key='spatial')
            asw = sdmbench.compute_ASW(adata_valid, 'pred', spatial_key='spatial')
            moranI, gearyC = sdmbench.marker_score(adata_valid, 'pred')
            
            # Store results in a dictionary
            result = {
                'Method': clustering_file_name,
                'NMI': nmi,
                'HOM': hom,
                'COM': com,
                'CHAOS': chaos,
                'PAS': pas,
                'ASW': asw,
                'Moran\'I': moranI,
                'Geary\'s C': gearyC
            }
            all_results.append(result)


