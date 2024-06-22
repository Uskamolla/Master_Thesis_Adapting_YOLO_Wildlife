import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

def get_vit_feature_extractor():
    model_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"  # Example ViT model from TFHub
    model = hub.KerasLayer(model_url, input_shape=(224, 224, 3))
    return model

def extract_features(image_paths, model):
    features = []
    for image_path in tqdm(image_paths, desc="Extracting features"):
        img_array = load_and_preprocess_image(image_path)
        feature_vector = model(tf.expand_dims(img_array, 0))
        features.append(np.squeeze(feature_vector))
    return np.array(features)

def save_features(features, file_path='extracted_features.npy'):
    np.save(file_path, features)

def load_images_from_folder(folder):
    image_paths = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image_paths.append(img_path)
    return image_paths

def main():
    image_folder = '/osm/uskamolla/Master_Thesis/clustering/cropped_datasets/cropped_wildboar_images'  # Update this path
    output_directory = '/osm/uskamolla/Master_Thesis/clustering/test/VTCC/wildboar_extracted_features'  # Update this path

    image_paths = load_images_from_folder(image_folder)
    vit_model = get_vit_feature_extractor()
    features = extract_features(image_paths, vit_model)

    if features.size == 0:
        raise ValueError("No features extracted. Please check the image paths and feature extraction process.")

    save_features(features, file_path=os.path.join(output_directory, 'extracted_features.npy'))

if __name__ == "__main__":
    main()
