import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, confusion_matrix
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


file_path = r'Z:\\coursework\\dataset\\cancer patient data sets.csv'
df = pd.read_csv(file_path)
true_labels = df['Level']
df_numerical = df.drop(['index', 'Patient Id', 'Level'], axis=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numerical)

silhouette_scores = []
davies_bouldin_scores = []
wcss = []
cluster_range = range(2, 9)

for n_clusters in cluster_range:
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agglomerative.fit_predict(df_scaled)

    cluster_centers = np.array([df_scaled[labels == i].mean(axis=0) for i in range(n_clusters)])
    distances = np.linalg.norm(df_scaled - cluster_centers[labels], axis=1)
    wcss.append(np.sum(distances ** 2))
    silhouette_scores.append(silhouette_score(df_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(df_scaled, labels))


plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")

plt.subplot(1, 3, 2)
plt.plot(cluster_range, davies_bouldin_scores, marker='o', color='red')
plt.title("Davies-Bouldin Index")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")

plt.subplot(1, 3, 3)
plt.plot(cluster_range, wcss, marker='o', color='green')
plt.title("Elbow Method (WCSS)")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.tight_layout()
plt.show()


optimal_clusters_silhouette = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on Silhouette Score: {optimal_clusters_silhouette}")
print(f"Best Silhouette Score: {max(silhouette_scores):.4f}")
optimal_clusters_db = cluster_range[np.argmin(davies_bouldin_scores)]
print(f"Optimal number of clusters based on Davies-Bouldin Index: {optimal_clusters_db}")
print(f"Best Davies-Bouldin Index: {min(davies_bouldin_scores):.4f}")



optimal_clusters = 4
agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
final_labels = agglomerative.fit_predict(df_scaled)
df['Cluster'] = final_labels
df[['Cluster']].head(10)



linkage_matrix = linkage(df_scaled, method='ward')
plt.figure(figsize=(16, 7))


plt.subplot(1, 2, 1)
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram without Cutoff')
plt.xlabel('Data Points')
plt.ylabel('Distance')

plt.subplot(1, 2, 2)
dendrogram(linkage_matrix, color_threshold=50)
plt.axhline(y=50, color='r', linestyle='--')
plt.title('Hierarchical Clustering Dendrogram with Cutoff at Distance 50')
plt.xlabel('Data Points')
plt.ylabel('Distance')

plt.tight_layout()
plt.show()



pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=final_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title("Agglomerative Clustering Results (4 Clusters) - PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()
print("Explained variance by each component:", pca.explained_variance_ratio_)



label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
true_labels_mapped = true_labels.map(label_mapping)
cm = confusion_matrix(true_labels_mapped, final_labels)
cluster_labels = [f'Cluster {i}' for i in range(optimal_clusters)]
true_class_labels = ['Low', 'Medium', 'High']
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix: True Labels vs. Predicted Clusters")
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
