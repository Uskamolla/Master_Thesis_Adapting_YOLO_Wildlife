import os
import numpy as np
from sklearn.manifold import TSNE
import hdbscan
import matplotlib.pyplot as plt
from shutil import copy2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import numpy as np
import os
from shutil import copy2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import umap


def process_images(features_path, output_folder):
    # Load pre-extracted features
    if os.path.exists(features_path):
        all_features = np.load(features_path)
        print(f"Loaded features from {features_path}")
    else:
        print("Features file not found. Please check the path.")
        return [], [], []

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30,random_state=0)
    # tsne = TSNE(n_components=2, perplexity= 150, n_iter=2000, random_state=0)

    print("Shape of the loaded features:", all_features.shape)

    # tsne = TSNE(n_components=2, perplexity=300, learning_rate=100, n_iter=2000, random_state=0)
    reduced_features_tsne = tsne.fit_transform(all_features)

    print("Shape of the reduced features:", reduced_features_tsne.shape)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=150, min_samples = 30)  # Adjust min_cluster_size as needed
    clusterer.fit(reduced_features_tsne)
    labels = clusterer.labels_

    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_features_tsne[:, 0], reduced_features_tsne[:, 1])
    plt.title('t-SNE visualization of image features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    tsne_plot_path = os.path.join(output_folder, 'tsne_visualization.png')
    plt.savefig(tsne_plot_path)
    plt.close()
    print(f"t-SNE visualization saved to: {tsne_plot_path}")
    # Visualization: t-SNE results
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_features_tsne[:, 0], reduced_features_tsne[:, 1], c=labels, cmap='Spectral', s=50)
    plt.title('t-SNE visualization with HDBSCAN clustering')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar()
    hdbscan_plot_path = os.path.join(output_folder, 'hdbscan_tsne_visualization.png')
    plt.savefig(hdbscan_plot_path)
    plt.close()
    print(f"HDBSCAN clustering visualization saved to: {hdbscan_plot_path}")

    return labels, reduced_features_tsne

def store_clustered_images(image_folder, labels, output_folder):
    # Create a folder for noise points if necessary
    noise_folder = os.path.join(output_folder, 'noise')
    os.makedirs(noise_folder, exist_ok=True)

    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    for img_path, label in zip(image_paths, labels):
        # If the label is -1, the point is considered noise by HDBSCAN
        if label == -1:
            destination = os.path.join(noise_folder, os.path.basename(img_path))
        else:
            cluster_folder = os.path.join(output_folder, f'cluster_{label}')
            os.makedirs(cluster_folder, exist_ok=True)
            destination = os.path.join(cluster_folder, os.path.basename(img_path))
        copy2(img_path, destination)

# Specify the path to the stored features and output folder
features_path = '/osm/uskamolla/Master_Thesis/clustering/test/VTCC/deer_extracted_features/extracted_features.npy'

output_folder = '/osm/uskamolla/Master_Thesis/clustering/test/VTCC/deer_clustering/HDBSCAN10'  # Modify this path
image_folder = '/osm/uskamolla/Master_Thesis/clustering/cropped_datasets/cropped_deer_images'
# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process the images using the stored features, perform t-SNE and HDBSCAN clustering
labels, reduced_features_tsne = process_images(features_path, output_folder)

# Store clustered images in respective directories
store_clustered_images(image_folder, labels, output_folder)
print(f"Clustered images are saved in: {output_folder}")


output_file_path = os.path.join(output_folder, 'hdbscan_parameters.txt')

# Open the file in write mode ('w') and write the information
with open(output_file_path, 'w') as file:
    # file.write("Unique labels: " + str(np.unique(cluster_labels)) + "\n")
    # file.write("Points labeled as noise: " + str(np.sum(cluster_labels == -1)) + "\n")
    file.write("    clusterer = hdbscan.HDBSCAN(min_cluster_size=150, min_samples = 30)  # Adjust min_cluster_size as needed")
    file.write("    tsne = TSNE(n_components=2,perplexity=30, random_state=0)")
    # file.write(" performed clustering on all_features")


