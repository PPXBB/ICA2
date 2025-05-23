# -*- coding: utf-8 -*-
"""
Created on Wed May 21 20:27:46 2025

@author: ppxbb
"""

import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp

from GraphST import GraphST

# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/root/miniconda3/envs/myconda/lib/R'

# Path where all datasets are stored
os.getcwd()
data_path = './DLPFC'  # Update the path to the folder where your datasets are stored
datasets = os.listdir(data_path)

# Output directory for saving images and text files
output_path = './output'
os.makedirs(output_path, exist_ok=True)
sc.settings.figdir = output_path

# Loop through each dataset folder
for dataset in datasets:
    dataset_path = os.path.join(data_path, dataset)
    
    # Check if it's a directory
    if os.path.isdir(dataset_path):
        print(f"Processing dataset: {dataset}")

        # Read the data
        adata = sc.read_visium(dataset_path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()

        # Add ground_truth
        df_meta = pd.read_csv(os.path.join(dataset_path, 'metadata.tsv'), sep='\t')
        df_meta_layer = df_meta['layer_guess']
        adata.obs['ground_truth'] = df_meta_layer.values

        # Get the number of clusters based on ground truth data
        n_clusters = df_meta_layer.nunique()

        # Define model
        model = GraphST.GraphST(adata, device=device)

        # Train the model
        adata = model.train()

        # Set radius for clustering
        radius = 50

        # Choose the clustering method
        tool = 'mclust'  # Change to 'leiden' or 'louvain' as needed
        if tool == 'mclust':
            from GraphST.utils import clustering
            clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
        elif tool in ['leiden', 'louvain']:
            clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

        # Filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        # Calculate ARI metric
        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
        adata.uns['ARI'] = ARI

        # # Print the ARI result
        # print(f"Dataset: {dataset}")
        # print(f"ARI: {ARI}")

        # Save the clustering results to a text file
        result_file = os.path.join(output_path, f"{dataset}_clustering_results.csv")
        adata.obs['domain'].to_csv(result_file, header=True, index=False)

        
        # Save the spatial clustering result as an image      
        sc.pl.spatial(adata,
                      img_key="hires",
                      color=["ground_truth", "domain"],
                      title=["Ground truth", f"ARI={ARI:.4f}"],
                      save=f"_{dataset}_clustering.png", show=False)


        print(f"Clustering results and images for {dataset} saved.")
