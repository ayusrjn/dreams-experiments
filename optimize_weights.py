import os
import pandas as pd
import numpy as np
import warnings
from PIL import Image

from combined_proximity import get_anyloc_features, get_clip_multimodal_features
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

def run_optimization(csv_path, image_dir):
    print("==========================================================")
    print(" DREAMS System: Hyperparameter Weight Optimizer           ")
    print("==========================================================")
    
    data = pd.read_csv(csv_path)
    img_paths = data['image_path'].tolist()
    captions = data['caption'].tolist()
    
    # 1. Extract features exactly once to save computational time
    print("\nExtracting Base Embeddings (This will only happen once)...")
    anyloc_feats = get_anyloc_features(img_paths, image_dir)
    clip_img_feats, _ = get_clip_multimodal_features(img_paths, captions, image_dir)
    
    print("\nComputing Base Similarity Matrices...")
    anyloc_sim = cosine_similarity(anyloc_feats)
    institutional_sim = cosine_similarity(clip_img_feats) # 100% CLIP Image Context
    
    valid_indices = [i for i in range(len(img_paths)) if not np.all(anyloc_feats[i] == 0)]
    
    best_acc = 0.0
    best_weights = []
    
    print(f"\nRunning Baseline Verification:")
    print(f" > Total Valid Images: {len(valid_indices)}")
    
    print("\nRunning High-Precision Grid Search over Weights (0.00 to 1.00)...")
    
    # Range of 0.0 to 1.0 inclusive with tiny 0.01 increments
    for w_anyloc in np.linspace(0.0, 1.0, 101):
        w_inst = 1.0 - w_anyloc
        
        # Combine the precomputed baseline similarities instantly
        final_proximity = (w_anyloc * anyloc_sim) + (w_inst * institutional_sim)
        
        matches = 0
        total = 0
        
        for i in valid_indices:
            sims_final = final_proximity[i].copy()
            sims_final[i] = -1.0 # exclude self-match
            best_idx = np.argmax(sims_final)
            
            true_class = img_paths[i].split('_')[0]
            matched_class = img_paths[best_idx].split('_')[0]
            
            if true_class == matched_class:
                matches += 1
            total += 1
            
        acc = (matches / total) * 100.0
        
        if acc > best_acc:
            best_acc = acc
            best_weights = [(w_anyloc, w_inst)]
        elif acc == best_acc:
            best_weights.append((w_anyloc, w_inst))
            
    print("\n" + "="*58)
    print(f" OPTIMIZATION COMPLETE - MAXIMUM DREAMS ACCURACY: {best_acc:.2f}%")
    print("="*58)
    print("\nThe following mathematically equivalent weight combinations hit this peak performance:\n")
    
    # Print out a nicely spaced sample of winning weights so the terminal isn't flooded
    step_size = max(1, len(best_weights) // 10)
    for w_a, w_i in best_weights[::step_size]:
        print(f"  > ANYLOC_WEIGHT: {w_a:.2f} | INSTITUTIONAL_WEIGHT: {w_i:.2f}")

    print(f"\nTotal exact combinations hitting {best_acc:.2f}%: {len(best_weights)}")
    
    max_anyloc_opt = best_weights[-1][0]
    print(f"\nPROPOSAL RECOMMENDATION:")
    print(f"I highly recommend selecting ANYLOC_WEIGHT = {max_anyloc_opt:.2f} (and INSTITUTIONAL = {1.0 - max_anyloc_opt:.2f}).")
    print("This combination gives you the absolute peak hybrid accuracy, while anchoring as physically strongly as possible to the objective/structural reality of the space.")

if __name__ == "__main__":
    run_optimization('dataset.csv', 'images')
