# DREAMS (Dynamic Recovery & Emotional Analysis Mapping System)
**Visual & Semantic Proximity Benchmarking for Clinical Contexts**

DREAMS is an experimental research architecture designed to mathematically prove **Functional Proximity**—how the structural and semantic realities of the physical spaces a patient frequents correlate with their psychological framing and emotional recovery.

Rather than relying on explicit geographic coordinates or brittle semantic labels, DREAMS utilizes modern self-supervised Vision Transformers (like DINOv2) to understand the *geometric archetypes* of an environment, bridged with Multimodal models (like CLIP) to understand the *cultural syntax* of the objects within it.

## The Core Concept
When a patient submits an unlabelled, "Instagram-tier" mobile photo (e.g., a snowy exterior, a dark bedroom, or a sterile waiting room), traditional computer vision categorizes it via explicit, rigid labels (e.g., classifying an icy hospital as a "ski resort").

DREAMS abandons rigid categorization. Instead, it measures distance against the patient's own historical spatial anchors using two fundamental metrics:
1. **Geometric Reality (AnyLoc / DINOv2):** Captures the pure structural shape, depth maps, and lighting boundaries of the environment, establishing un-biased physical Functional Proximity.
2. **Semantic Reality (CLIP Vision):** Identifies human-recognizable structural components (e.g., "this is a cluster of tables" = Restaurant) without relying on text.

By fusing these modalities, the engine perfectly overcomes blinding environmental noise (snow, harsh fluorescence, weird camera angles) to cluster the patient's life into functional domains.

---

## The Experiments & Benchmarks
This repository contains benchmark scripts evaluating several computer vision methodologies against the DREAMS Hybrid theory:

### 1. `combined_proximity.py` (The DREAMS Engine)
The definitive hybrid proximity engine. Fuses the structural anchors of DINOv2 (AnyLoc) with the object-level semantic mapping of CLIP.
* Computes weighted visual topologies.
* Reaches peak **92.6%+ Functional Domain Accuracy**.
* Generates `hybrid_mds_clusters.png` (2D topological map) and `hybrid_dendrogram.png` (Hierarchical merging tree).

### 2. `optimize_weights.py` (Hyperparameter Tuning)
Mathematical optimizer utilizing Grid Search to verify the optimal blending ratio of Geometric vs Semantic vision across 100+ simulated epochs. Proved that over-indexing on semantic language causes "Emotional Hallucinations" (e.g., grouping a Park with a Hospital), establishing the baseline `0.80 To 0.20` AnyLoc/CLIP anchor schema.

### 3. `anyloc_clustering.py` & `clip_clustering.py` (The Baselines)
Standalone modules that evaluate the raw isolation of their respective methodologies.
* **AnyLoc:** Acts as the Spatial Geometric baseline.
* **CLIP:** Acts as the Cultural/Object Institutional baseline.

### 4. `anyloc_superpoint.py` (Identity Proximity)
A two-stage hierarchical verification pipeline. Uses the rapid global descriptor (AnyLoc) to find Functional candidates, and then deploys a robust local geometric matcher (proxy via SIFT/RANSAC) to calculate inlier points. Proves whether a patient returned to the exact physical room/space under different spatial conditions or times.

### 5. `places365.py`
A demonstration script utilizing a supervised ResNet-50. Highlights the catastrophic failures of explicit categorical learning on clinical data (e.g., classifying a hospital in Alaska as a `parking_garage` or `ski_resort`).

### 6. Legacy Comparisons
Scripts evaluating the DREAMS architecture against older, rigid Visual Place Recognition (VPR) paradigms:
* `cosplace_vs_anyloc.py`
* `netvlad_vs_anyloc.py`
* `clip_vs_anyloc.py`

---

## Installation & Environment
Ensure you have PyTorch configured for your specific compute environment before proceeding.

```bash
# Clone the repository
git clone https://github.com/ayusrjn/dreams-experiments.git
cd dreams-experiments

# Install required dependencies
pip install -r requirements.txt
```

## Running the Engine
Execute the definitive clustering benchmark to map the current dataset into the four functional vectors (Hospital, Park, Residential, Restaurant):
```bash
python combined_proximity.py
```
