import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Use PyTorch to extract DINOv2 visual features (AnyLoc uses DINO features)
try:
    import torch
    import torchvision.transforms as T
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def extract_visual_features(img_paths, image_dir):
    """
    Extracts high-dimensional visual features from images using DINOv2.
    This is the core of AnyLoc's approach: using DINOv2 for Visual Place Recognition
    to find if park images naturally group with other parks structurally.
    """
    if not HAS_TORCH:
        print("Error: PyTorch and Scikit-Learn are required for actual image feature extraction.")
        return np.random.rand(len(img_paths), 384) # DINOv2 ViT-S outputs 384-dim
        
    print("Loading AnyLoc Foundation Model (DINOv2 ViT-S/14) for Feature Extraction...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
    except Exception as e:
        print(f"Error loading DINOv2 from torch.hub: {e}")
        print("Fallback to random features.")
        return np.random.rand(len(img_paths), 384)
    
    # DINOv2 standard preprocessing
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    features = []
    print(f"Extracting DINOv2 visual features for {len(img_paths)} images in '{image_dir}'...")
    
    for filename in img_paths:
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            print(f"  Warning: Image not found {path}, using zero vector.")
            features.append(np.zeros(384))
            continue
            
        try:
            img = Image.open(path).convert('RGB')
            # Extract only the 3 RGB channels (handles potential RGBA alpha issues safely)
            img_tensor = preprocess(img)[:3].unsqueeze(0)
            
            with torch.no_grad():
                # Extract CLS token from DINOv2
                feat = model(img_tensor).squeeze().numpy()
            features.append(feat)
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            features.append(np.zeros(384))
            
    return np.array(features)

def experiment_anyloc_visual(csv_path, image_dir):
    print(f"--- DREAMS System: Visual Spatial Classification (Image-Based) ---")
    data = pd.read_csv(csv_path)
    
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} directory not found.")
        return
        
    # 1. Extract true visual descriptors (pixel-based)
    visual_features = extract_visual_features(data['image_path'].tolist(), image_dir)
    
    if HAS_TORCH:
        print("\nCalculating Visual Similarity Matrix (Cosine Similarity)...")
        sim_matrix = cosine_similarity(visual_features)
        
        print("\n[Layer A] Visual Similarities (Top matches purely based on pixels):")
        # For each image, find its most visually similar image
        matched_classes = 0
        total_valid = 0
        
        for i, row in data.iterrows():
            img_name = row['image_path']
            # Ignore zero-vectors (missing images)
            if np.all(visual_features[i] == 0):
                continue
                
            sims = sim_matrix[i].copy()
            sims[i] = -1.0 # Exclude self-match
            
            best_match_idx = np.argmax(sims)
            best_match_name = data.iloc[best_match_idx]['image_path']
            best_match_score = sims[best_match_idx]
            
            # Ground truth class from filename
            true_class_1 = img_name.split('_')[0]
            true_class_2 = best_match_name.split('_')[0]
            
            is_match = true_class_1 == true_class_2
            status = "MATCH!" if is_match else "different"
            if is_match:
                matched_classes += 1
            total_valid += 1
                
            print(f"  {img_name} \n  -> Visually Similar To: {best_match_name} (Sim: {best_match_score:.2f}) [{status}]")
            
        acc = (matched_classes / total_valid) * 100 if total_valid > 0 else 0
        print(f"\nVisual Classification Accuracy (Categorical grouping): {acc:.1f}%")
        
        # 2. PCA Visualization mapping structural properties to 2D
        print("\nGenerating Visual Feature Space Mapping (PCA)...")
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(visual_features)
        
        plt.figure(figsize=(12, 8))
        
        # Assign colors based on implicit ground truth to see if they form clusters
        data['true_class'] = data['image_path'].apply(lambda x: x.split('_')[0].capitalize())
        classes = data['true_class'].unique()
        colors = {'Hospital': '#e6194b', 'Park': '#3cb44b', 'Residential': '#ffe119', 'Restaurant': '#4363d8'}
        
        for cls in classes:
            indices = data.index[data['true_class'] == cls].tolist()
            valid_indices = [idx for idx in indices if not np.all(visual_features[idx] == 0)]
            
            if valid_indices:
                plt.scatter(reduced_features[valid_indices, 0], reduced_features[valid_indices, 1], 
                            label=cls, s=150, alpha=0.8, color=colors.get(cls, 'gray'), edgecolor='white')
                
                # Annotate points
                for idx in valid_indices:
                    short_name = data.iloc[idx]['image_path'].split('.')[0]
                    plt.annotate(short_name, (reduced_features[idx, 0] + 0.2, reduced_features[idx, 1]), 
                                 fontsize=8, alpha=0.9)
                                 
        plt.title('AnyLoc Image-Based Evaluation: Visual Proximity Clusters', fontsize=16)
        plt.xlabel('Principal Component 1 (Visual Geometry)', fontsize=12)
        plt.ylabel('Principal Component 2 (Lighting/Texture)', fontsize=12)
        plt.legend(title="Functional Class")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visual_clusters.png', dpi=300)
        print("Saved purely visual cluster map to 'visual_clusters.png'")
        
if __name__ == "__main__":
    experiment_anyloc_visual('dataset.csv', 'images')
