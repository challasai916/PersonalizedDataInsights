import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st

def find_optimal_clusters(df, max_clusters=10):
    """
    Find the optimal number of clusters using the silhouette score.
    
    Parameters:
    df (DataFrame): Data for clustering
    max_clusters (int): Maximum number of clusters to try
    
    Returns:
    int: Optimal number of clusters
    dict: Silhouette scores for each number of clusters
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = {}
    
    # Start from 2 clusters (silhouette score is not defined for 1 cluster)
    for n_clusters in range(2, min(max_clusters + 1, len(df))):
        # Skip if we have too few samples for the number of clusters
        if len(df) <= n_clusters:
            continue
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score
        try:
            score = silhouette_score(scaled_data, labels)
            silhouette_scores[n_clusters] = score
        except:
            # Handle potential errors (e.g., only one sample in a cluster)
            silhouette_scores[n_clusters] = 0
    
    # Find the number of clusters with the highest silhouette score
    if not silhouette_scores:
        # If we couldn't calculate any scores, default to 3 clusters
        return 3, {}
    
    optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
    
    return optimal_clusters, silhouette_scores


def perform_segmentation(df, n_clusters=None, max_clusters=10):
    """
    Perform user segmentation using KMeans clustering.
    
    Parameters:
    df (DataFrame): Data for segmentation
    n_clusters (int): Number of clusters (if None, find optimal)
    max_clusters (int): Maximum number of clusters to try
    
    Returns:
    tuple: (cluster_labels, cluster_centers, silhouette_scores, pca_result, pca_explained_variance)
    """
    if df is None or df.empty:
        st.error("No data available for segmentation.")
        return None, None, None, None, None
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Determine the number of clusters if not provided
    if n_clusters is None:
        n_clusters, silhouette_scores = find_optimal_clusters(df, max_clusters)
    else:
        silhouette_scores = {}  # Empty if not calculated
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # Get cluster centers (in the original feature space)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Perform PCA for visualization
    # Use min(n_features, 2) components
    n_components = min(df.shape[1], 2)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    return labels, centers, silhouette_scores, pca_result, pca.explained_variance_ratio_


def describe_segments(df, labels, feature_cols):
    """
    Generate descriptive statistics for each segment.
    
    Parameters:
    df (DataFrame): Original data
    labels (array): Cluster labels
    feature_cols (list): Feature column names
    
    Returns:
    DataFrame: Segment profiles
    """
    if df is None or df.empty or labels is None:
        st.error("No data available for segment description.")
        return None
    
    # Add segment labels to the dataframe
    df_with_labels = df.copy()
    df_with_labels['Segment'] = labels
    
    # Calculate mean values for each feature in each segment
    segment_profiles = df_with_labels.groupby('Segment')[feature_cols].mean()
    
    # Calculate overall means for comparison
    overall_means = df[feature_cols].mean()
    
    # Calculate how each segment compares to the overall mean
    segment_comparisons = segment_profiles.copy()
    for col in feature_cols:
        segment_comparisons[col] = segment_profiles[col] / overall_means[col]
    
    # Prepare the results
    results = {
        'profiles': segment_profiles,
        'comparisons': segment_comparisons,
        'counts': df_with_labels['Segment'].value_counts().sort_index()
    }
    
    return results
