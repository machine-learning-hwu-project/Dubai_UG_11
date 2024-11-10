import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# List of file paths for the datasets with shortened labels
file_paths = {
    '1_ef_b': r'Z:\\coursework\\datasets\\1_binary\\processed\\1_ef_b.csv',
    '1_nrml_b': r'Z:\\coursework\\datasets\\1_binary\\processed\\1_nrml_b.csv',
    '1_ef_ub': r'Z:\\coursework\\datasets\\1_binary\\processed\\1_ef_ub.csv',
    '1_nrml_ub': r'Z:\\coursework\\datasets\\1_binary\\processed\\1_nrml_ub.csv'
}

# Dictionary to store results
results = {}

# Loop through each dataset
for label, file_path in file_paths.items():
    # Load dataset
    df = pd.read_csv(file_path)
    true_labels = df['LUNG_CANCER']  # Assuming 'LUNG_CANCER' is the target column
    df_numerical = df.drop(['LUNG_CANCER'], axis=1, errors='ignore')  # Adjust as needed

    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerical)

    # Cluster Evaluation
    silhouette_scores = []
    davies_bouldin_scores = []
    wcss = []
    cluster_range = range(2, 6)

    for n_clusters in cluster_range:
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agglomerative.fit_predict(df_scaled)

        # Calculate cluster metrics
        cluster_centers = np.array([df_scaled[labels == i].mean(axis=0) for i in range(n_clusters)])
        distances = np.linalg.norm(df_scaled - cluster_centers[labels], axis=1)
        wcss.append(np.sum(distances ** 2))
        silhouette_scores.append(silhouette_score(df_scaled, labels))
        davies_bouldin_scores.append(davies_bouldin_score(df_scaled, labels))

    # Store metrics for the current dataset
    results[label] = {
        "Silhouette Scores": silhouette_scores,
        "Davies-Bouldin Scores": davies_bouldin_scores,
        "WCSS": wcss,
        "Best Silhouette Score": max(silhouette_scores),
        "Best Davies-Bouldin Index": min(davies_bouldin_scores),
        "Optimal Clusters (Silhouette)": cluster_range[np.argmax(silhouette_scores)],
        "Optimal Clusters (Davies-Bouldin)": cluster_range[np.argmin(davies_bouldin_scores)]
    }

    # Perform final clustering with optimal cluster number 4
    optimal_clusters = 4
    agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
    final_labels = agglomerative.fit_predict(df_scaled)
    df['Cluster'] = final_labels

    # Generate linkage matrix for dendrogram visualization
    linkage_matrix = linkage(df_scaled, method='ward')

    # Plot dendrograms with and without cutoff
    plt.figure(figsize=(16, 7))

    # Dendrogram without cutoff
    plt.subplot(1, 2, 1)
    dendrogram(linkage_matrix)
    plt.title(f'{label} - Dendrogram without Cutoff')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')

    # Dendrogram with cutoff at distance 50
    plt.subplot(1, 2, 2)
    dendrogram(linkage_matrix, color_threshold=50)
    plt.axhline(y=50, color='r', linestyle='--')
    plt.title(f'{label} - Dendrogram with Cutoff at Distance 50')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')

    plt.tight_layout()
    plt.show()

    # PCA visualization of clusters
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=final_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f"{label} - Agglomerative Clustering Results (4 Clusters) - PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster')
    plt.show()

    # Print explained variance by each component
    print(f"{label} - Explained variance by each component:", pca.explained_variance_ratio_)

    # Generate and display confusion matrix to compare true labels with predicted clusters
    cm = confusion_matrix(true_labels, final_labels)
    cluster_labels = [f'Cluster {i}' for i in range(optimal_clusters)]
    true_class_labels = ['No Cancer', 'Cancer']  # Adjust as per your true label categories

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"{label} - Confusion Matrix: True Labels vs. Predicted Clusters")
    plt.colorbar()
    plt.xticks(np.arange(optimal_clusters), cluster_labels, rotation=45)
    plt.yticks(np.arange(len(true_class_labels)), true_class_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")

    plt.xlabel("Predicted Clusters")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()
