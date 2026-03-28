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
    from transformers import CLIPProcessor, CLIPModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_clip_multimodal_features(img_paths, captions, image_dir):
    print("Loading CLIP (openai/clip-vit-base-patch32) for Multimodal Cultural Context...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        return np.random.rand(len(img_paths), 512), np.random.rand(len(img_paths), 512)
        
    img_features = []
    text_features = []
    print("  Extracting CLIP visual and semantic (caption) features (Institutional Proximity)...")
    
    for filename, text in zip(img_paths, captions):
        path = os.path.join(image_dir, filename)
        
        # 1. Text (Caption) Encoding
        try:
            if pd.isna(text) or text.strip() == "":
                text = "a photograph of a location"
                
            text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                t_out = model.get_text_features(**text_inputs)
                if not isinstance(t_out, torch.Tensor):
                    t_out = t_out.pooler_output if hasattr(t_out, 'pooler_output') else t_out[0]
                t_feat = t_out.squeeze().numpy()
            text_features.append(t_feat)
        except Exception as e:
            print(f"Error on text features for {filename}: {e}")
            text_features.append(np.zeros(512))
            
        # 2. Image Encoding
        if not os.path.exists(path):
            img_features.append(np.zeros(512))
            continue
            
        try:
            img = Image.open(path).convert('RGB')
            img_inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                i_out = model.get_image_features(**img_inputs)
                if not isinstance(i_out, torch.Tensor):
                    i_out = i_out.pooler_output if hasattr(i_out, 'pooler_output') else i_out[0]
                i_feat = i_out.squeeze().numpy()
            img_features.append(i_feat)
        except Exception as e:
            print(f"Error on image features for {filename}: {e}")
            img_features.append(np.zeros(512))
            
    return np.array(img_features), np.array(text_features)

def run_clip_clustering(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: Baseline Proximity Engine                 ")
    print(" Pure CLIP (Multimodal Cultural/Institutional ONLY)       ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    if not HAS_TORCH:
        print("Error: PyTorch and Transformers required.")
        return
        
    img_paths = data['image_path'].tolist()
    captions = data['caption'].tolist()
    
    # 1. Extract Embeddings
    print("\n[Phase 1] Evaluating Semantic Institutional Reality")
    clip_img_feats, clip_text_feats = get_clip_multimodal_features(img_paths, captions, image_dir)
    
    # 2. Compute Similarity
    clip_img_sim = cosine_similarity(clip_img_feats)
    clip_text_sim = cosine_similarity(clip_text_feats)
    institutional_sim = (clip_img_sim + clip_text_sim) / 2.0
    
    # 3. Clustering Module
    print("\n" + "="*58)
    print(" Running Clustering on Baseline CLIP Proximity Matrix")
    print("="*58)
    
    try:
        valid_indices = [idx for idx in range(len(img_paths)) if not np.all(clip_img_feats[idx] == 0)]
        valid_proximity = institutional_sim[np.ix_(valid_indices, valid_indices)]
        valid_names = [img_paths[idx].split('.')[0] for idx in valid_indices]
        
        valid_proximity = np.clip(valid_proximity, 0.0, 1.0)
        distance_matrix = 1.0 - valid_proximity
        
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
    
    # Compute MDS
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)
        
        plt.figure(figsize=(12, 8))
        # Magma used distinctly to separate cultural mapping from the AnyLoc structural mapping
        plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='magma', s=150, alpha=0.8, edgecolor='white')
        
        for idx, name in enumerate(valid_names):
            plt.annotate(name, (coords[idx, 0] + 0.015, coords[idx, 1] + 0.015), fontsize=9)
            
        plt.title('DREAMS Semantic Topography (MDS over Pure CLIP/Cultural Proximity)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('clip_mds_clusters.png', dpi=300)
        print(" > Saved 'clip_mds_clusters.png'")
    except Exception as e:
        print(f"MDS Mapping failed: {e}")
    
    # Compute Dendrogram
    try:
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='ward')
        
        plt.figure(figsize=(14, 10))
        dendrogram(Z, labels=valid_names, leaf_rotation=90, leaf_font_size=11)
        plt.title('Hierarchical Proximity Tree (Pure Multimodal Extractor)', fontsize=14)
        plt.ylabel('Distance (1.0 - CLIP Institutional/Cultural Proximity Score)')
        plt.tight_layout()
        plt.savefig('clip_dendrogram.png', dpi=300)
        print(" > Saved 'clip_dendrogram.png'")
    except Exception as e:
        print(f"Dendrogram generation failed: {e}")

if __name__ == "__main__":
    run_clip_clustering('dataset.csv', 'images')
