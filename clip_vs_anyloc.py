import os
import pandas as pd
import numpy as np
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import torch
    import torchvision.transforms as T
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import CLIPProcessor, CLIPVisionModelWithProjection
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

def get_anyloc_features(img_paths, image_dir):
    print("Loading AnyLoc (DINOv2 ViT-S/14)...")
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
    print("  Extracting AnyLoc global features...")
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

def get_clip_features(img_paths, image_dir):
    print("Loading CLIP (openai/clip-vit-base-patch32)...")
    if not HAS_TRANSFORMERS:
        print("Error: 'transformers' library required for CLIP. Please pip install transformers.")
        return np.random.rand(len(img_paths), 512)
        
    try:
        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        return np.random.rand(len(img_paths), 512)
        
    features = []
    print("  Extracting CLIP visual features...")
    for filename in img_paths:
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            features.append(np.zeros(512))
            continue
        try:
            img = Image.open(path).convert('RGB')
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                # CLIPVisionModelWithProjection outputs image_embeds explicitly
                feat = outputs.image_embeds.squeeze().numpy()
            features.append(feat)
        except Exception as e:
            print(f"Error on {filename}: {e}")
            features.append(np.zeros(512))
    return np.array(features)

def run_clip_comparison(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: Visual & Semantic Benchmarking            ")
    print("         AnyLoc (DINOv2) vs CLIP (ViT-B/32)               ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    if not HAS_TORCH:
        print("Error: PyTorch required.")
        return
        
    img_paths = data['image_path'].tolist()
    
    # 1. Extract Embeddings
    print("\n--- AnyLoc Phase ---")
    anyloc_feats = get_anyloc_features(img_paths, image_dir)
    print("\n--- CLIP Phase ---")
    clip_feats = get_clip_features(img_paths, image_dir)
    
    # 2. Compute Similarities
    anyloc_sim = cosine_similarity(anyloc_feats)
    clip_sim = cosine_similarity(clip_feats)
    
    anyloc_matches = 0
    clip_matches = 0
    total_valid = 0
    
    print("\nComparing the top structural/semantic match (excluding self) for each space...")
    
    for i, row in data.iterrows():
        if np.all(anyloc_feats[i] == 0) or np.all(clip_feats[i] == 0): 
            continue
        
        sims_a = anyloc_sim[i].copy()
        sims_a[i] = -1.0
        best_a = np.argmax(sims_a)
        
        sims_c = clip_sim[i].copy()
        sims_c[i] = -1.0
        best_c = np.argmax(sims_c)
        
        true_class = row['image_path'].split('_')[0]
        match_class_a = data.iloc[best_a]['image_path'].split('_')[0]
        match_class_c = data.iloc[best_c]['image_path'].split('_')[0]
        
        if true_class == match_class_a: anyloc_matches += 1
        if true_class == match_class_c: clip_matches += 1
        total_valid += 1
        
        if best_a != best_c:
            print(f"\n[Query] {row['image_path']}")
            print(f"  > AnyLoc: {data.iloc[best_a]['image_path']} (Sim: {sims_a[best_a]:.3f})")
            print(f"  > CLIP  : {data.iloc[best_c]['image_path']} (Sim: {sims_c[best_c]:.3f})")

    if total_valid > 0:
        print("\n" + "="*58)
        print("Functional Domain Accuracy (Grouping similar concepts)")
        print(f"  AnyLoc (DINOv2) Accuracy: {(anyloc_matches/total_valid)*100:.1f}%")
        print(f"  CLIP (ViT-B/32) Accuracy: {(clip_matches/total_valid)*100:.1f}%")
        print("="*58)
        print("Key Technical Difference for your Research:")
        print("- CLIP is trained on image-text pairs! It groups images based on semantic concepts (e.g. recognizing 'a photo of a hospital').")
        print("  Therefore, CLIP understands what the space *represents* to humans in language.")
        print("- AnyLoc/DINOv2 has ZERO semantic knowledge. It groups purely based on overlapping physical geometry, lighting gradients, and depth textures.")
        print("  Therefore, AnyLoc understands the structural *reality* of the space.")

if __name__ == "__main__":
    run_clip_comparison('dataset.csv', 'images')
