import os
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def download_places365_files():
    # Labels
    labels_file = 'categories_places365.txt'
    if not os.path.exists(labels_file):
        print(f"Downloading {labels_file}...")
        url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        urllib.request.urlretrieve(url, labels_file)
        
    classes = list()
    with open(labels_file) as class_file:
        for line in class_file:
            # Classes look like "/a/airport_terminal", we strip the first 3 chars
            label_name = line.strip().split(' ')[0][3:]
            classes.append(label_name)
    classes = tuple(classes)
    
    # Weights
    weight_file = 'resnet50_places365.pth.tar'
    if not os.path.exists(weight_file):
        print(f"Downloading {weight_file} (~90MB)...")
        url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        urllib.request.urlretrieve(url, weight_file)
        
    return classes, weight_file

def load_places365_model(weight_file):
    print("Loading ResNet-50 Places365 Model...")
    # ResNet50 configured for 365 scene classes
    model = models.resnet50(num_classes=365)
    
    checkpoint = torch.load(weight_file, map_location=lambda storage, loc: storage)
    # The weights might be saved from DataParallel, so remove 'module.' prefix
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def analyze_dataset_places(csv_path, image_dir):
    print("--- DREAMS System: Places365 Explicit Scene Recognition ---")
    
    if not HAS_TORCH:
        print("Error: PyTorch required for Places365 evaluation.")
        return
        
    data = pd.read_csv(csv_path)
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        return
        
    classes, weight_file = download_places365_files()
    model = load_places365_model(weight_file)
    
    # Standard Places365 Image Transformations
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    
    print("\nPredicting explicit Places365 Categories for dataset images...\n")
    
    for _, row in data.iterrows():
        img_name = row['image_path']
        path = os.path.join(image_dir, img_name)
        
        # Ground Truth Functional Mapping
        true_class = img_name.split('_')[0].capitalize()
        
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            results.append({
                'image': img_name, 
                'true_class': true_class, 
                'pred1': 'N/A', 'pred2': 'N/A', 'pred3': 'N/A'
            })
            continue
            
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                
            probs = torch.nn.functional.softmax(outputs, 1)[0] * 100
            _, indices = torch.sort(outputs, descending=True)
            
            # Extract top 3 categories and probabilities
            top3_cats = [classes[idx] for idx in indices[0][:3]]
            top3_probs = [probs[idx].item() for idx in indices[0][:3]]
            
            results.append({
                'image': img_name,
                'true_class': true_class,
                'pred1': f"{top3_cats[0]} ({top3_probs[0]:.1f}%)",
                'pred2': f"{top3_cats[1]} ({top3_probs[1]:.1f}%)",
                'pred3': f"{top3_cats[2]} ({top3_probs[2]:.1f}%)",
            })
            
            print(f"[{true_class}] {img_name}")
            print(f"  -> Explicit Match: 1. {top3_cats[0]} ({top3_probs[0]:.1f}%), 2. {top3_cats[1]} ({top3_probs[1]:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            
    # Export explicitly predicted categories for comparison
    df_results = pd.DataFrame(results)
    df_results.to_csv('places365_results.csv', index=False)
    print("\nSaved full explicit scene predictions to 'places365_results.csv'")
    
    print("\nThis contrasts with AnyLoc's *implicit* matching, which forces purely structural visual proximity rather than rigid categorical labels!")

if __name__ == "__main__":
    analyze_dataset_places('dataset.csv', 'images')
