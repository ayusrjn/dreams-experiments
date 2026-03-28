import os
import pandas as pd
import numpy as np
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------
# 1. Classic NetVLAD Architecture Implementation
# ---------------------------------------------------------
class NetVLAD(nn.Module):
    """
    Standard NetVLAD layer for Visual Place Recognition (VPR).
    Aggregates local CNN spatial features into a compact global descriptor.
    """
    def __init__(self, num_clusters=16, dim=2048, alpha=100.0):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        
        # Convolutional layer for Soft-Assignment
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        # Cluster centroids
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        N, C = x.shape[:2]
        
        # Soft-assignment of spatial features to clusters
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        x_flatten = x.view(N, C, -1)
        
        # Calculate residuals to each cluster centroid
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C_idx in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[C_idx:C_idx+1, :].view(1, -1, 1)
            residual *= soft_assign[:, C_idx:C_idx+1, :].unsqueeze(2)
            vlad[:, C_idx:C_idx+1, :] = residual.sum(dim=-1)
            
        # Intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2)  
        # Flatten
        vlad = vlad.view(x.size(0), -1)       
        # Final L2 normalizations
        vlad = F.normalize(vlad, p=2, dim=1)  
        return vlad

class CNN_NetVLAD(nn.Module):
    def __init__(self):
        super(CNN_NetVLAD, self).__init__()
        # Backbone: ResNet50 up to the last convolutional layer
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Drop the AvgPool and FC layers to keep the 2D spatial feature map (B x 2048 x 7 x 7)
        layers = list(resnet.children())[:-2] 
        self.backbone = nn.Sequential(*layers)
        
        # NetVLAD pooling layer on top
        self.netvlad = NetVLAD(num_clusters=16, dim=2048)

    def forward(self, x):
        x = self.backbone(x) # Extract spatial features
        x = self.netvlad(x)  # Aggregate via NetVLAD
        return x

# ---------------------------------------------------------
# 2. Feature Extraction Modules
# ---------------------------------------------------------
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

def get_netvlad_features(img_paths, image_dir):
    print("Loading CNN + NetVLAD Architecture...")
    model = CNN_NetVLAD()
    model.eval()
    
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    print("  Extracting NetVLAD pooled features...")
    for filename in img_paths:
        path = os.path.join(image_dir, filename)
        if not os.path.exists(path):
            features.append(np.zeros(2048 * 16)) # 16 clusters x 2048 dim = 32768
            continue
        try:
            img = Image.open(path).convert('RGB')
            with torch.no_grad():
                feat = model(preprocess(img)[:3].unsqueeze(0)).squeeze().numpy()
            features.append(feat)
        except Exception as e:
            print(f"Error on {filename}: {e}")
            features.append(np.zeros(2048 * 16))
    return np.array(features)

# ---------------------------------------------------------
# 3. Execution & Benchmarking
# ---------------------------------------------------------
def run_netvlad_comparison(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: VPR Algorithm Benchmarking                ")
    print("         AnyLoc (DINOv2) vs CNN+NetVLAD                   ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    if not HAS_TORCH:
        print("Error: PyTorch required.")
        return
        
    img_paths = data['image_path'].tolist()
    
    # 1. Extract VPR Embeddings
    print("\n--- AnyLoc Phase ---")
    anyloc_feats = get_anyloc_features(img_paths, image_dir)
    print("\n--- NetVLAD Phase ---")
    netvlad_feats = get_netvlad_features(img_paths, image_dir)
    
    # 2. Compute Similarities
    anyloc_sim = cosine_similarity(anyloc_feats)
    netvlad_sim = cosine_similarity(netvlad_feats)
    
    anyloc_matches = 0
    netvlad_matches = 0
    total_valid = 0
    
    print("\nComparing the top structural match (excluding self) for each space...")
    
    for i, row in data.iterrows():
        if np.all(anyloc_feats[i] == 0) or np.all(netvlad_feats[i] == 0): 
            continue
        
        sims_a = anyloc_sim[i].copy()
        sims_a[i] = -1.0
        best_a = np.argmax(sims_a)
        
        sims_n = netvlad_sim[i].copy()
        sims_n[i] = -1.0
        best_n = np.argmax(sims_n)
        
        true_class = row['image_path'].split('_')[0]
        match_class_a = data.iloc[best_a]['image_path'].split('_')[0]
        match_class_n = data.iloc[best_n]['image_path'].split('_')[0]
        
        if true_class == match_class_a: anyloc_matches += 1
        if true_class == match_class_n: netvlad_matches += 1
        total_valid += 1
        
        if best_a != best_n:
            print(f"\n[Query] {row['image_path']}")
            print(f"  > AnyLoc  : {data.iloc[best_a]['image_path']} (Sim: {sims_a[best_a]:.3f})")
            print(f"  > NetVLAD : {data.iloc[best_n]['image_path']} (Sim: {sims_n[best_n]:.3f})")

    if total_valid > 0:
        print("\n" + "="*58)
        print("Functional Domain Accuracy (Grouping similar concepts)")
        print(f"  AnyLoc (DINOv2) Accuracy        : {(anyloc_matches/total_valid)*100:.1f}%")
        print(f"  CNN+NetVLAD (Untrained Baseline): {(netvlad_matches/total_valid)*100:.1f}%")
        print("="*58)
        print("Key Technical Difference:")
        print("- NetVLAD aggregates local CNN features into a massive cluster histogram (here 32,768 dimensions).")
        print("  (Our pipeline demonstrates the pure architectural behavior before expensive geo-supervised training).")
        print("- AnyLoc completely bypasses CNNs, directly utilizing DINOv2's robust self-supervised similarities.")

if __name__ == "__main__":
    run_netvlad_comparison('dataset.csv', 'images')
