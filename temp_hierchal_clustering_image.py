import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.decomposition import PCA

# Define the path to the dataset and subdirectories
base_path = r'Z:\coursework\dataset\lung_images\Training cases'
case_types = ['Benign cases', 'Malignant cases', 'Normal cases']
true_labels = []  # Store true labels for evaluation
images = []       # Store image data
sample_images = {}  # Dictionary to store a sample of processed images

# Step 1: Load and preprocess images
for label, case_type in enumerate(case_types):
    folder_path = os.path.join(base_path, case_type)
    sample_images[case_type] = []  # Initialize list for each case type

    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # Resize for consistency
            
            if i < 3:  # Save the first three images from each folder for display
                sample_images[case_type].append(img)
            
            images.append(img.flatten())       # Flatten to turn image into a feature vector
            true_labels.append(label)          # Label images based on folder (0 for Benign, 1 for Malignant, 2 for Normal)

# Step 2: Display sample processed images from each category
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
fig.suptitle("Sample Processed Images from Each Category")

for row, (case_type, imgs) in enumerate(sample_images.items()):
    for col, img in enumerate(imgs):
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        # Add label below each image
        axes[row, col].set_title(case_type, fontsize=10, pad=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Step 3: Create a DataFrame and standardize features
df_images = pd.DataFrame(images)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_images)


# Step 4: Apply Agglomerative Clustering for 3 clusters
linkage_matrix = linkage(df_scaled, method='ward')
cluster_labels = cut_tree(linkage_matrix, n_clusters=3).flatten()

# Add cluster labels to the original data
df_images['Cluster'] = cluster_labels


df_images.head()


# Step 3: Create a DataFrame and standardize features
df_images = pd.DataFrame(images)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_images)
df_scaled.head()

# Step 4: Apply Agglomerative Clustering for 3 clusters
linkage_matrix = linkage(df_scaled, method='ward')
cluster_labels = cut_tree(linkage_matrix, n_clusters=3).flatten()

# Add cluster labels to the original data
df_images['Cluster'] = cluster_labels

# Step 5: Visualize the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, color_threshold=1350)  # Adjust threshold if needed
plt.axhline(y=1350, color='r', linestyle='--')
plt.title('Hierarchical Clustering Dendrogram with 3 Clusters')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Step 7: Apply PCA for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Plot the clusters using the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title("Agglomerative Clustering Results (3 Clusters) - PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

# Print explained variance by each component
print("Explained variance by each component:", pca.explained_variance_ratio_)




# Silhouette and Davies-Bouldin Scores
silhouette_avg = silhouette_score(df_scaled, cluster_labels)
db_score = davies_bouldin_score(df_scaled, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Index: {db_score}')

# Visualize Confusion Matrix
# Step 6: Evaluate Clustering Performance
# Confusion matrix
cm = confusion_matrix(true_labels, cluster_labels)
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix: True Labels vs. Predicted Clusters")
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Benign', 'Malignant', 'Normal'], rotation=45)
plt.yticks(tick_marks, ['Benign', 'Malignant', 'Normal'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")

plt.xlabel("Predicted Clusters")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()
