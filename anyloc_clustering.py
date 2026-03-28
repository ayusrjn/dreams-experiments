import os
import pandas as pd
import numpy as np
import warnings
from PIL import Image
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')

try:
    import torch
    import torchvision.transforms as T
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_anyloc_features(img_paths, image_dir):
    print("Loading AnyLoc Foundation Model (DINOv2 ViT-S/14)...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model.eval()
    except Exception as e:
        print(f"Error loading DINOv2: {e}")
        return np.random.rand(len(img_paths), 384)
        
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    features = []
    print("  Extracting pure AnyLoc structural features (Functional Proximity)...")
    
    for filename in img_paths:
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            features.append(np.zeros(384))
            continue
        try:
            img = Image.open(path).convert('RGB')
            with torch.no_grad():
                feat = model(preprocess(img)[:3].unsqueeze(0)).squeeze().numpy()
            features.append(feat)
        except Exception:
            features.append(np.zeros(384))
    return np.array(features)

def run_anyloc_clustering(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: Baseline Proximity Engine                 ")
    print(" Pure AnyLoc (Structural/Geometric ONLY)                  ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    if not HAS_TORCH:
        print("Error: PyTorch required.")
        return
        
    img_paths = data['image_path'].tolist()
    
    # 1. Extract Embeddings
    print("\n[Phase 1] Evaluating Robust Spatial Reality (No Semantic Labels)")
    anyloc_feats = get_anyloc_features(img_paths, image_dir)
    
    # 2. Compute Similarity
    anyloc_sim = cosine_similarity(anyloc_feats)
    
    # 3. Clustering Module
    print("\n" + "="*58)
    print(" Running Clustering on Baseline AnyLoc Proximity Matrix")
    print("="*58)
    
    # Filter out empty rows (zero vectors from missing images)
    try:
        valid_indices = [idx for idx in range(len(img_paths)) if not np.all(anyloc_feats[idx] == 0)]
        valid_proximity = anyloc_sim[np.ix_(valid_indices, valid_indices)]
        valid_names = [img_paths[idx].split('.')[0] for idx in valid_indices]
        
        # Clip and convert to Dis-similarity (Distance) matrix
        valid_proximity = np.clip(valid_proximity, 0.0, 1.0)
        distance_matrix = 1.0 - valid_proximity
        
        # Symmetrize and zero-diagonal to prevent float precision errors in linkage
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0)
        
        print("\n[Phase 2] Spectral Clustering (n=4)")
        spectral = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
        cluster_labels = spectral.fit_predict(valid_proximity)
        
        for c_id in range(4):
            members = [valid_names[k] for k, lbl in enumerate(cluster_labels) if lbl == c_id]
            print(f"\nCluster {c_id} ({len(members)} environments):")
            for m in members:
                # Show true grouping class
                prefix = m.split('_')[0].upper()
                print(f"  [{prefix}] {m}")
    except Exception as e:
        print(f"Spectral Clustering failed: {e}")
        cluster_labels = np.zeros(len(valid_names))
            
    print("\n[Phase 3] Generating Topological Proximity Maps...")
    
    # Compute MDS (2D topological mapping of distances)
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)
        
        plt.figure(figsize=(12, 8))
        # Use a distinct colormap from the hybrid map to differentiate at a glance
        plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='viridis', s=150, alpha=0.8, edgecolor='k')
        
        for idx, name in enumerate(valid_names):
            plt.annotate(name, (coords[idx, 0] + 0.015, coords[idx, 1] + 0.015), fontsize=9)
            
        plt.title('DREAMS Baseline Topography (MDS over Pure AnyLoc Proximity)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('anyloc_mds_clusters.png', dpi=300)
        print(" > Saved 'anyloc_mds_clusters.png'")
    except Exception as e:
        print(f"MDS Mapping failed: {e}")
    
    # Compute Dendrogram (Hierarchical Tree)
    try:
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='ward')
        
        plt.figure(figsize=(14, 10))
        dendrogram(Z, labels=valid_names, leaf_rotation=90, leaf_font_size=11)
        plt.title('Hierarchical Proximity Tree (Pure AnyLoc Vision Baseline)', fontsize=14)
        plt.ylabel('Distance (1.0 - AnyLoc Proximity Score)')
        plt.tight_layout()
        plt.savefig('anyloc_dendrogram.png', dpi=300)
        print(" > Saved 'anyloc_dendrogram.png'")
    except Exception as e:
        print(f"Dendrogram generation failed: {e}")

if __name__ == "__main__":
    run_anyloc_clustering('dataset.csv', 'images')
