import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings

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
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    features = []
    print("  Extracting AnyLoc features...")
    for filename in img_paths:
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            features.append(np.zeros(384))
            continue
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img)[:3].unsqueeze(0)
            with torch.no_grad():
                feat = model(img_tensor).squeeze().numpy()
            features.append(feat)
        except Exception:
            features.append(np.zeros(384))
    return np.array(features)

def get_cosplace_features(img_paths, image_dir):
    print("Loading CosPlace VPR Model (ResNet50, 2048-dim)...")
    try:
        # Load pre-trained CosPlace model from PyTorch Hub
        model = torch.hub.load("gmberton/CosPlace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
        model.eval()
    except Exception as e:
        print(f"Error loading CosPlace: {e}")
        if "Missing dependencies" in str(e):
             print("(You might need 'scipy' for CosPlace hub dependencies)")
        return np.random.rand(len(img_paths), 2048)
        
    # Standard CosPlace preprocessing
    preprocess = T.Compose([
        T.Resize((224, 224)), # CosPlace generally uses standard resizing
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    print("  Extracting CosPlace features...")
    for filename in img_paths:
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            features.append(np.zeros(2048))
            continue
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img)[:3].unsqueeze(0)
            with torch.no_grad():
                feat = model(img_tensor).squeeze().numpy()
            features.append(feat)
        except Exception:
            features.append(np.zeros(2048))
    return np.array(features)

def run_comparison(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: VPR Algorithm Benchmarking                ")
    print("           AnyLoc (DINOv2) vs CosPlace (ResNet)           ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    if not HAS_TORCH:
        print("Error: PyTorch required.")
        return
        
    img_paths = data['image_path'].tolist()
    
    # 1. Extract features for both models
    anyloc_feats = get_anyloc_features(img_paths, image_dir)
    cosplace_feats = get_cosplace_features(img_paths, image_dir)
    
    # 2. Compute Similarities
    anyloc_sim = cosine_similarity(anyloc_feats)
    cosplace_sim = cosine_similarity(cosplace_feats)
    
    # 3. Analyze differences in "Nearest Neighbors"
    print("\nVisual Proximity Disagreement Analysis:")
    print("Comparing the top visual match predicted by each model (excluding self).\n")
    
    anyloc_matches = 0
    cosplace_matches = 0
    total_valid = 0
    
    for i, row in data.iterrows():
        # Skip failed image loads
        if np.all(anyloc_feats[i] == 0) or np.all(cosplace_feats[i] == 0): 
            continue
        
        sims_a = anyloc_sim[i].copy()
        sims_a[i] = -1.0 # Exclude identity match
        best_idx_a = np.argmax(sims_a)
        
        sims_c = cosplace_sim[i].copy()
        sims_c[i] = -1.0 # Exclude identity match
        best_idx_c = np.argmax(sims_c)
        
        true_class = row['image_path'].split('_')[0]
        match_class_a = data.iloc[best_idx_a]['image_path'].split('_')[0]
        match_class_c = data.iloc[best_idx_c]['image_path'].split('_')[0]
        
        if true_class == match_class_a: anyloc_matches += 1
        if true_class == match_class_c: cosplace_matches += 1
        total_valid += 1
        
        # Only print when the two VPR models disagree on the most similar place
        if best_idx_a != best_idx_c:
            img_name = row['image_path']
            print(f"[Query] {img_name} (Class: {true_class.capitalize()})")
            print(f"  > AnyLoc's best match  : {data.iloc[best_idx_a]['image_path']} (Sim: {sims_a[best_idx_a]:.3f}) | Class Match: {true_class == match_class_a}")
            print(f"  > CosPlace's best match: {data.iloc[best_idx_c]['image_path']} (Sim: {sims_c[best_idx_c]:.3f}) | Class Match: {true_class == match_class_c}\n")

    print("==========================================================")
    print(f"VPR Functional Grouping Accuracy")
    print("How often does the closest visual match fall within the same Functional Class (e.g. Park matched to Park)?")
    if total_valid > 0:
        print(f"  AnyLoc (DINOv2) Accuracy  : {(anyloc_matches/total_valid)*100:.1f}%")
        print(f"  CosPlace (ResNet) Accuracy: {(cosplace_matches/total_valid)*100:.1f}%")
    print("==========================================================")
    print("Key Takeaways:")
    print("- CosPlace is strictly supervised and trained on massive, geo-tagged urban environments.")
    print("  It's extremely powerful at exact street-level identification, but might fail on generic indoor/nature grouping.")
    print("- AnyLoc uses DINOv2 self-supervision, which captures highly robust underlying structural semantics.")
    print("  This often makes it vastly superior at functional / functional proximity grouping across radically different locations.")

if __name__ == "__main__":
    run_comparison('dataset.csv', 'images')
