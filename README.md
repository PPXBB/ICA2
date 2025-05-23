# Spatial Transcriptomics Clustering Benchmark and Analysis

This repository contains code to perform spatial transcriptomics analysis using different clustering techniques, including **GraphST**, **Louvain**, and **stLearn**, and evaluate their performance on publicly available datasets. The aim is to benchmark the performance of these clustering algorithms on spatial transcriptomics data, using various metrics like **NMI**, **ARI**, **HOM**, **COM**, and others.

## Overview

The project is organized into three main modules:

1. **GraphST Clustering** (`graphst_all.py`): Implements the **GraphST** method for spatial transcriptomics clustering using graph neural networks.
2. **STLearn Clustering** (`stlearn_all.py`): Implements clustering using the **STLearn** package, which integrates spatial information and gene expression for clustering.
3. **Evaluation** (`evaluation.py`): Handles the benchmarking of clustering results using multiple metrics (e.g., **NMI**, **HOM**, **COM**).. 

These scripts are designed to work with datasets such as the **DLPFC** (Dorsolateral Prefrontal Cortex) and support multiple clustering methods.

## Requirements

- **Python 3.8+**
- **PyTorch**
- **Scanpy**
- **STLearn**
- **GraphST** 
- **Pandas**
- **Numpy**
- **Scikit-learn**
- **Matplotlib**
- **Squidpy**
- **SDMBench**
- **R** 

## Getting Started

### 1. **GraphST Clustering**

The `graphst_all.py` script applies the **GraphST** method to spatial transcriptomics data using graph neural networks. It reads datasets, applies clustering, and generates spatial clustering results and images.

#### Usage:

```bash
python graphst_all.py
```

### 2. **STLearn Clustering**

The `stlearn_all.py` script processes spatial transcriptomics data, applies clustering methods (Louvain, stLearn), and saves the results. It uses **STLearn** for spatial information integration.

#### Usage:

```bash
python stlearn_all.py
```

### 3. **Evaluation**

The `evaluation.py` script benchmarks clustering results using various evaluation metrics (e.g., **NMI**, **HOM**, **COM**). It compares the performance of **GraphST**, **Louvain**, and **STLearn** on the given datasets.

#### Usage:

```bash
python evaluation.py
```

## Dataset

The datasets used in the analysis are publicly available spatial transcriptomics data. You can modify the `data_path` variable in the scripts to point to your local dataset directory.

- The **DLPFC** dataset is located in the `./DLPFC` directory (or set your path).
- Metadata files are expected to be in TSV format, containing the ground truth annotations for different layers in the tissue.

## Output

The results of clustering and the evaluation metrics will be saved in the `./output` directory, where each dataset's results are stored in CSV format. Additionally, spatial plots of the clustering results are saved as images.

## Evaluation Metrics

The evaluation scripts compute various clustering metrics:

- **Normalized Mutual Information (NMI)**
- **Homogeneity (HOM)**
- **Completeness (COM)**
- **CHAOS**
- **Adjusted Rand Index (ARI)**
- **Average Silhouette Width (ASW)**
- **Moran's I**
- **Geary's C**
