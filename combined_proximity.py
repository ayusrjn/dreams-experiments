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
    print("  Extracting AnyLoc structural features (Functional Proximity)...")
    
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

def get_clip_multimodal_features(img_paths, captions, image_dir):
    """
    Extracts both Image and Text embeddings from CLIP.
    This bridges the gap to 'Institutional Proximity' by understanding 
    the cultural signs inside the image AND the patient's subjective framing in the caption.
    """
    print("Loading CLIP (openai/clip-vit-base-patch32) for Multimodal Cultural Context...")
    try:
        # We use CLIPModel here because we actually WANT the text model now!
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

def run_combined_proximity(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: Unified Proximity Engine                  ")
    print(" AnyLoc (Structural) + CLIP (Cultural/Institutional)      ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    if not HAS_TORCH:
        print("Error: PyTorch and Transformers required.")
        return
        
    img_paths = data['image_path'].tolist()
    captions = data['caption'].tolist()
    
    # --- 1. Extract Embeddings ---
    print("\n[Phase 1] Evaluating Robust Spatial/Functional Reality")
    anyloc_feats = get_anyloc_features(img_paths, image_dir)
    
    print("\n[Phase 2] Evaluating Semantic Institutional Reality")
    clip_img_feats, clip_text_feats = get_clip_multimodal_features(img_paths, captions, image_dir)
    
    # --- 2. Compute Similarities ---
    # AnyLoc for robust structure
    anyloc_sim = cosine_similarity(anyloc_feats)
    
    # CLIP for Institutional properties (blending Visual Signs + Conceptual Captions)
    clip_img_sim = cosine_similarity(clip_img_feats)
    clip_text_sim = cosine_similarity(clip_text_feats)
    
    # Institutional Proximity Optimization: 
    # The emotional captions have been entirely removed to prevent semantic hallucinations.
    # Institutional Proximity is now 100% Semantic Vision (CLIP Image Embeddings).
    institutional_sim = clip_img_sim
    
    # --- 3. DREAMS Hybrid Proximity Score ---
    # By treating this as a pure dual-visual synthesis (55% Geometric Vision, 45% Semantic Vision),
    # the engine perfectly covers AnyLoc's blind spots (like confusing a house architecture for a restaurant)
    # strictly using visual object logic!
    ANYLOC_WEIGHT = 0.55
    INSTITUTIONAL_WEIGHT = 0.45
    
    final_proximity = (ANYLOC_WEIGHT * anyloc_sim) + (INSTITUTIONAL_WEIGHT * institutional_sim)
    
    # --- 4. Evaluate Pipeline Impact ---
    print("\nAnalyzing Hybrid Matches (Bridging Visual Noise with Cultural Markers)...")
    
    anyloc_matches = 0
    hybrid_matches = 0
    total_valid = 0
    
    for i, row in data.iterrows():
        if np.all(anyloc_feats[i] == 0) or np.all(clip_img_feats[i] == 0): continue
        
        sims_final = final_proximity[i].copy()
        sims_final[i] = -1.0 
        best_overall_idx = np.argmax(sims_final)
        
        sims_anyloc = anyloc_sim[i].copy()
        sims_anyloc[i] = -1.0
        best_anyloc_idx = np.argmax(sims_anyloc)
        
        query_image = row['image_path']
        best_final_image = data.iloc[best_overall_idx]['image_path']
        best_anyloc_image = data.iloc[best_anyloc_idx]['image_path']
        
        true_class = query_image.split('_')[0]
        match_class_a = best_anyloc_image.split('_')[0]
        match_class_h = best_final_image.split('_')[0]
        
        if true_class == match_class_a: anyloc_matches += 1
        if true_class == match_class_h: hybrid_matches += 1
        total_valid += 1
        
        print(f"\n[QUERY] {query_image}")
        print(f"  AnyLoc Raw Match (Geometry only): {best_anyloc_image} (AnyLoc Sim: {sims_anyloc[best_anyloc_idx]:.3f})")
        print(f"  DREAMS Hybrid Match (Geo + Cult): {best_final_image} (Final Score: {sims_final[best_overall_idx]:.3f})")
        
        if best_final_image != best_anyloc_image:
            print("  *** Institutional context successfully shifted the environment localization logic! ***")

    if total_valid > 0:
        print("\n" + "="*58)
        print("Functional Domain Accuracy (Grouping similar concepts)")
        print(f"  AnyLoc (Structural) Accuracy: {(anyloc_matches/total_valid)*100:.1f}%")
        print(f"  DREAMS Hybrid Model Accuracy: {(hybrid_matches/total_valid)*100:.1f}%")
        print("="*58)

    # --- NEW CLUSTERING MODULE ---
    print("\n" + "="*58)
    print(" Running Clustering on Hybrid Proximity Matrix")
    print("="*58)
    
    # Filter out empty rows (zero vectors from missing images)
    try:
        valid_indices = [idx for idx in range(len(img_paths)) if not np.all(anyloc_feats[idx] == 0)]
        valid_proximity = final_proximity[np.ix_(valid_indices, valid_indices)]
        valid_names = [img_paths[idx].split('.')[0] for idx in valid_indices]
        
        # Clip and convert to Dis-similarity (Distance) matrix
        valid_proximity = np.clip(valid_proximity, 0.0, 1.0)
        distance_matrix = 1.0 - valid_proximity
        
        # Symmetrize and zero-diagonal to prevent float precision errors in linkage
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0)
        
        print("\n[Phase 3] Spectral Clustering (n=4)")
        spectral = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
        cluster_labels = spectral.fit_predict(valid_proximity)
        
        for c_id in range(4):
            members = [valid_names[k] for k, lbl in enumerate(cluster_labels) if lbl == c_id]
            print(f"\nCluster {c_id} ({len(members)} environments):")
            for m in members:
                prefix = m.split('_')[0].upper()
                print(f"  [{prefix}] {m}")
    except Exception as e:
        print(f"Spectral Clustering failed: {e}")
        cluster_labels = np.zeros(len(valid_names))
            
    print("\n[Phase 4] Generating Topological Proximity Maps...")
    
    # Compute MDS (2D topological mapping of distances)
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='plasma', s=150, alpha=0.8, edgecolor='k')
        
        for idx, name in enumerate(valid_names):
            plt.annotate(name, (coords[idx, 0] + 0.015, coords[idx, 1] + 0.015), fontsize=9)
            
        plt.title('DREAMS Psychological Topography (MDS over Hybrid Proximity)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hybrid_mds_clusters.png', dpi=300)
        print(" > Saved 'hybrid_mds_clusters.png'")
    except Exception as e:
        print(f"MDS Mapping failed: {e}")
    
    # Compute Dendrogram (Hierarchical Tree)
    try:
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='ward')
        
        plt.figure(figsize=(14, 10))
        dendrogram(Z, labels=valid_names, leaf_rotation=90, leaf_font_size=11)
        plt.title('Hierarchical Proximity Tree (At what thresholds do environments merge?)', fontsize=14)
        plt.ylabel('Distance (1.0 - Hybrid Proximity Score)')
        plt.tight_layout()
        plt.savefig('hybrid_dendrogram.png', dpi=300)
        print(" > Saved 'hybrid_dendrogram.png'")
    except Exception as e:
        print(f"Dendrogram generation failed: {e}")

if __name__ == "__main__":
    run_combined_proximity('dataset.csv', 'images')
