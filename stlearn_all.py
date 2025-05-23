# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:50:22 2025

@author: ppxbb
"""
import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy

# import module
import stlearn as st
from pathlib import Path
st.settings.set_figure_params(dpi=180)


# specify PATH to data
BASE_PATH = Path("D:/code/CMML/ICA2/Data/DLPFC")


# here we include all 12 samples
sample_list = ["151507", "151508", "151509",
               "151510", "151669", "151670",
               "151671", "151672", "151673",
               "151674", "151675", "151676"]


for i in range(len(sample_list)):
    sample = sample_list[i]

    GROUND_TRUTH_PATH = BASE_PATH / sample / "metadata.tsv"
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH, sep='\t', index_col=0)
    # ground_truth_df.index = ground_truth_df.index.map(lambda x: x[7:])

    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df["layer_guess"].values))
    ground_truth_df["ground_truth_le"] = ground_truth_le

    # load data
    data = st.Read10X(BASE_PATH / sample)
    ground_truth_df = ground_truth_df.reindex(data.obs_names)
    data.obs["ground_truth"] = pd.Categorical(ground_truth_df["layer_guess"])
    st.pl.cluster_plot(data, use_label="ground_truth", cell_alpha=0.5)


for i in range(12):
    sample = sample_list[i]
    GROUND_TRUTH_PATH = BASE_PATH / sample / "metadata.tsv"
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH, sep='\t', index_col=0)

    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df["layer_guess"].values))
    ground_truth_df["ground_truth_le"] = ground_truth_le
    TILE_PATH = Path("./tmp/{}_tiles".format(sample))
    TILE_PATH.mkdir(parents=True, exist_ok=True)
    data = st.Read10X(BASE_PATH / sample)
    ground_truth_df = ground_truth_df.reindex(data.obs_names)
    n_cluster = len((set(ground_truth_df["layer_guess"]))) - 1
    data.obs['ground_truth'] = ground_truth_df["layer_guess"]
    ground_truth_le = ground_truth_df["ground_truth_le"]
    
    data = data[np.logical_not(data.obs['ground_truth'].isna())]#remove NAN
    
    # pre-processing for gene count table
    st.pp.filter_genes(data,min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)

    # run PCA for gene expression data
    st.em.run_pca(data,n_comps=15)
    
    st.pp.neighbors(data)
    st.tl.clustering.louvain(data, resolution=0.8)
    st.pl.cluster_plot(data,use_label="louvain")

    # pre-processing for spot image
    st.pp.tiling(data, TILE_PATH)

    # this step uses deep learning model to extract high-level features from tile images
    # may need few minutes to be completed
    st.pp.extract_feature(data)

    # stSME
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']

    st.pp.scale(data_)
    st.em.run_pca(data_,n_comps=15)

    st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")
    st.pl.cluster_plot(data_, use_label="X_pca_kmeans")
    
    
    output_path = 'D:/code/CMML/ICA2/benchmark/output/stlearn'
    result_file_st = os.path.join(output_path, f"{sample}_clustering_results.csv")
    data_.obs['X_pca_kmeans'].to_csv(result_file_st, header=True, index=False)
        
    output_path = 'D:/code/CMML/ICA2/benchmark/output/louvain'
    result_file_lou = os.path.join(output_path, f"{sample}_clustering_results.csv")
    data_.obs['louvain'].to_csv(result_file_lou, header=True, index=False)
    





