import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
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

def extract_global_features(img_paths, image_dir):
    """
    Step 1: AnyLoc Global Proximity (DINOv2)
    Analyzes visual structure to test functional similarity.
    """
    if not HAS_TORCH:
        print("Error: PyTorch required for global feature extraction.")
        return np.random.rand(len(img_paths), 384)
        
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
    print(f"Extracting global AnyLoc features for {len(img_paths)} images...")
    
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
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            features.append(np.zeros(384))
            
    return np.array(features)

def match_local_features(img1_path, img2_path):
    """
    Step 2: SuperPoint Identity Match Proxy (SIFT/ORB)
    Extracts local keypoints to determine if it's the exact same physical room/location.
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0, None, None, None, None, None
        
    # Standardize scale for consistent feature matching
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    
    # Initialize highly robust local feature detector
    # using SIFT as a reliable out-of-the-box structural substitute for SuperPoint
    sift = cv2.SIFT_create(nfeatures=2000)
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0, None, None, None, None, None

    # FLANN fast approximate nearest neighbors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test to filter valid geometric matches vs noise
    good_matches = []
    for m_n in matches:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    # Use RANSAC robust estimation to filter out outlier lines
    if len(good_matches) > 10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist() if mask is not None else []
        inliers = sum(matchesMask)
        # Keep only RANSAC inliers for rendering
        filtered_matches = [m for i, m in enumerate(good_matches) if matchesMask[i]]
    else:
        M, matchesMask = None, []
        inliers = 0
        filtered_matches = good_matches
        
    return inliers, img1, kp1, img2, kp2, filtered_matches

def visualize_match(img1, kp1, img2, kp2, matches, name1, name2, inliers):
    """
    Draws the local geometric feature connections bridging two images.
    """
    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green
                       singlePointColor = None,
                       flags = 2) # Flag 2 means draw only matched points
                       
    # Draw limited top matches to prevent drawing a massive illegible web
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, **draw_params)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    
    title = f"IDENTITY VERIFIED: Same Physical Location Confirmed\n" \
            f"SuperPoint Proxy Found {inliers} Keypoint Inliers\n" \
            f"Comparing: {name1} VS {name2}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    out_name = f"identity_match_{name1.split('.')[0]}_{name2.split('.')[0]}.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    print(f"Saved geometric visualization to '{out_name}'")

def run_anyloc_superpoint_pipeline(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: Two-Stage Proximity Resolution            ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    
    if not HAS_TORCH:
        print("Error: PyTorch required for global tracking.")
        return
        
    # Phase 1: High Level Functional Proximity
    features = extract_global_features(data['image_path'].tolist(), image_dir)
    sim_matrix = cosine_similarity(features)
    
    GLOBAL_SIMILARITY_THRESHOLD = 0.50
    LOCAL_INLIER_THRESHOLD = 15 # Required robust keypoints to guarantee identical room
    
    print(f"\n[PHASE 1] AnyLoc Global Perception Pass (Threshold > {GLOBAL_SIMILARITY_THRESHOLD})")
    candidate_pairs = []
    
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            sim = sim_matrix[i, j]
            # Verify they are actually finding zero-vectors vs true images
            if np.all(features[i] == 0) or np.all(features[j] == 0):
                continue
            if sim > GLOBAL_SIMILARITY_THRESHOLD:
                candidate_pairs.append((i, j, sim))
                
    print(f"Identified {len(candidate_pairs)} candidate pairs passing AnyLoc.")
    
    best_match_viz = None
    max_inliers = 0
    
    print(f"\n[PHASE 2] SuperPoint Identity Verification Pass")
    
    for idx1, idx2, sim in candidate_pairs:
        img1_name = data.iloc[idx1]['image_path']
        img2_name = data.iloc[idx2]['image_path']
        
        path1 = os.path.join(image_dir, img1_name)
        path2 = os.path.join(image_dir, img2_name)
        
        # Verify specific structural geometry
        inliers, img1, kp1, img2, kp2, matches = match_local_features(path1, path2)
        
        print(f"\nEvaluating -> {img1_name} & {img2_name}")
        print(f"  AnyLoc DINO Score : {sim:.3f}")
        print(f"  SuperPoint Inliers: {inliers}")
        
        if inliers > LOCAL_INLIER_THRESHOLD:
            print("  [✓] IDENTITY PROXIMITY CONFIRMED (Same physical location)")
            # Track the strongest geographical match for generation
            if inliers > max_inliers:
                max_inliers = inliers
                best_match_viz = (img1, kp1, img2, kp2, matches, img1_name, img2_name)
        else:
            print("  [X] FUNCTIONAL PROXIMITY ONLY (Similar aesthetic/category, different physical geometry)")
            
    print("\n" + "="*58)
    
    if best_match_viz is not None:
        print(f"\nGenerating 'Identity Proximity' graphic for the highest-confidence match...")
        visualize_match(best_match_viz[0], best_match_viz[1], best_match_viz[2], 
                        best_match_viz[3], best_match_viz[4], best_match_viz[5], 
                        best_match_viz[6], max_inliers)
    else:
        print("\nNo identity matches exceeded the rigorous keypoint threshold.")

if __name__ == "__main__":
    run_anyloc_superpoint_pipeline('dataset.csv', 'images')
