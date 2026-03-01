# Automatic Feature Matching Across Images



---

## Overview

This project implements an automatic feature matching pipeline between pairs of images using **Harris Corner Detection** and the **Nearest Neighbor Distance Ratio (NNDR)** method. Given two images of the same scene taken from slightly different viewpoints, the pipeline detects salient keypoints, describes them using local patch descriptors, and matches them across images.

---

## Pipeline

```
Harris Corner Detection → Non-Maximal Suppression → Patch Descriptor Extraction → NNDR Matching → Match Visualization
```

---

## Methodology

### Step 1: Harris Corner Detection
Interest points are detected using the Harris corner detector. The algorithm computes the second-moment matrix from image gradients and identifies corners as points with high curvature in all directions.

### Step 2: Non-Maximal Suppression (NMS)
A 20×20 pixel sliding window is used to retain only local maxima of the Harris response, eliminating redundant nearby detections.

### Step 3: Feature Descriptor Extraction
For each detected corner, a 40×40 pixel patch is extracted and downsampled to an 8×8 patch (64-dimensional descriptor) using Gaussian anti-aliasing before downsampling to improve robustness.

### Step 4: Feature Matching via NNDR
For each descriptor in Image 1, the two nearest neighbors in Image 2 are found using **SSD (Sum of Squared Differences)**. A match is accepted only if:

```
NNDR = distance(1NN) / distance(2NN) < threshold (0.5)
```

A low NNDR means the best match is significantly better than the second-best, indicating a distinctive and reliable match.

---

## Parameters

| Parameter | Value |
|---|---|
| Harris sigma | 1.5 |
| Harris threshold | 10% of max response |
| NMS window size | 20×20 px |
| Descriptor patch size | 40×40 → 8×8 (64-dim) |
| Anti-aliasing | Enabled |
| Similarity metric | SSD |
| NNDR threshold | 0.5 |

---

## Sample Results (Scene 1)

### Input Image Pair
> *(Place your two input images side by side here)*
> Example: `scene1/img1.jpg` and `scene1/img2.jpg`

---

### Step 1 — Harris Corners Detected
Red dots mark all detected corners before suppression.

![Harris Corners](scene1/1_corners_detected.png)

---

### Step 2 — After Non-Maximal Suppression
Only the most salient, locally-maximal corners are retained.

![After NMS](scene1/2_corners_nms.png)

---

### Step 3 — NNDR Histogram
The red dashed line marks the NNDR threshold of **0.5**. Features to the left are accepted as valid matches; features to the right are rejected as ambiguous.

![NNDR Histogram](scene1/3_nndr_histogram.png)

---

### Step 4 — Top 5 Best Matches (1NN vs 2NN)
Each row shows: the query feature patch from Image 1 | its 1st nearest neighbor in Image 2 | its 2nd nearest neighbor in Image 2. Lower NNDR = more distinctive match.

![Top 5 Matches](scene1/4_top_matches.png)

---

### Step 5 — Match Visualization

**Option A — Line Connections:** Green lines connect matched features. Red dots mark unmatched (ambiguous) features.

![Line Matches](scene1/5_matches_lines.png)

**Option B — Color-Coded Matches:** Each matched pair shares a color across both images, with a numbered index for reference.

![Color Coded Matches](scene1/6_matches_colored.png)

---

## Key Observations

- The NNDR threshold of 0.5 effectively separates distinctive matches (low ratio) from ambiguous ones (high ratio), visible as a bimodal distribution in the histogram.
- Harris corners cluster around high-contrast regions like building edges, window corners, and architectural details.
- The 8×8 downsampled patch descriptors with anti-aliasing strike a good balance between descriptor distinctiveness and computational efficiency.
- The 1NN patches are visually similar to their query features while 2NN patches are clearly different, validating the NNDR approach.

---

## Dependencies

```bash
pip install opencv-python numpy scikit-image matplotlib
```

---

## Usage

```bash
python solution.py
```

Output images for each processing step are saved to the respective scene folder (`scene1/`, `scene2/`). Open `index.html` in a browser to view the full results report.

---
