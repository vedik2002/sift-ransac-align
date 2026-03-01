import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, peak_local_max
from scipy.ndimage import maximum_filter
from skimage.transform import resize
from matplotlib.patches import ConnectionPatch
import os

class FeatureMatcher:

    def __init__(self,image1_path,image2_path):
        self.img1 = cv2.imread(image1_path)
        self.img2 = cv2.imread(image2_path)

        self.gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
    
        self.response1 = None
        self.response2 = None
        self.corners1 = None
        self.corners2 = None
        self.corners1_nms = None
        self.corners2_nms = None

        self.descriptors1 = None
        self.descriptors2 = None



    def harris_corner_detection(self, sigma=1.5, min_distance=10, edge_discard=20):
       
        def detect_harris(gray_image):
            # skimage expects float image
            gray_float = gray_image.astype(np.float32)

            # Compute Harris response
            h = corner_harris(gray_float, method='eps', sigma=sigma)

            # Extract local maxima
            threshold = 0.10 * h.max()
            coords = np.argwhere(h > threshold)

            # Remove corners near edges
            h_img, w_img = gray_image.shape
            mask = (
                (coords[:, 0] > edge_discard) &
                (coords[:, 0] < h_img - edge_discard) &
                (coords[:, 1] > edge_discard) &
                (coords[:, 1] < w_img - edge_discard)
            )
            coords = coords[mask]

            return coords, h

        self.corners1, self.response1 = detect_harris(self.gray1)
        self.corners2, self.response2 = detect_harris(self.gray2)

        print(f"Image 1: Detected {len(self.corners1)} Harris corners")
        print(f"Image 2: Detected {len(self.corners2)} Harris corners")

        return self.corners1, self.corners2
    
    def visualize_corners(self, output_path='corners_detected.png'):
        """Visualize detected Harris corners"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
   
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img1_rgb)
        axes[0].plot(self.corners1[:, 1], self.corners1[:, 0], 'r.', markersize=2)
        axes[0].set_title(f'Image 1 - Harris Corners ({len(self.corners1)} points)')
        axes[0].axis('off')
        
     
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        axes[1].imshow(img2_rgb)
        axes[1].plot(self.corners2[:, 1], self.corners2[:, 0], 'r.', markersize=2)
        axes[1].set_title(f'Image 2 - Harris Corners ({len(self.corners2)} points)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
        


    def non_maximal_suppression(self, window_size=20):
        """
        Step 2: Non-Maximal Suppression (NMS)
        
        Args:
            window_size: Size of window for local maximum detection
        
        Returns:
            corners1_nms, corners2_nms: Arrays of suppressed corner coordinates
        """
        def apply_nms(corners, response, window_size):
        
            local_max = maximum_filter(response, size=window_size)
            
          
            is_max = (response == local_max)
            
           
            corners_nms = []
            for y, x in corners:
                if is_max[y, x]:
                    corners_nms.append([y, x])
            
            return np.array(corners_nms)
        
        self.corners1_nms = apply_nms(self.corners1, self.response1, window_size)
        self.corners2_nms = apply_nms(self.corners2, self.response2, window_size)
        
        print(f"After NMS (window_size={window_size}):")
        print(f"  Image 1: {len(self.corners1_nms)} corners")
        print(f"  Image 2: {len(self.corners2_nms)} corners")
        
        return self.corners1_nms, self.corners2_nms
    
    def extract_feature_descriptors(self,patch_size=40,descriptor_size=8):


        def extract_descriptors(gray_image,corners,patch_size,descriptor_size):
            descriptors = []
            valid_corners = []
            half_patch = patch_size // 2
            h,w = gray_image.shape


            for y,x in corners:
                if (y - half_patch < 0 or y + half_patch >= h or
                    x - half_patch < 0 or x + half_patch >= w):
                    continue

                patch = gray_image[y - half_patch:y + half_patch, x - half_patch:x + half_patch]
                patch_resized = resize(patch, (descriptor_size, descriptor_size), anti_aliasing=True,mode='reflect')
                
                mean = patch_resized.mean()
                std = patch_resized.std()
                if std > 1e-7:
                    patch_normalized = (patch_resized - mean) / std
                else:
                    patch_normalized = patch_resized - mean
                
                descriptor = patch_normalized.flatten()
                descriptors.append(descriptor)
                valid_corners.append([y,x])
            
            return np.array(descriptors), np.array(valid_corners)
        
        self.descriptors1, valid_corners1 = extract_descriptors(self.gray1,self.corners1_nms,patch_size,descriptor_size)
        self.descriptors2, valid_corners2 = extract_descriptors(self.gray2,self.corners2_nms,patch_size,descriptor_size)
        
        self.corners1_nms = valid_corners1
        self.corners2_nms = valid_corners2

        print(f"Extracted {len(self.descriptors1)} descriptors from image 1 (8x8 = 64 dimensions each)")
        print(f"Extracted {len(self.descriptors2)} descriptors from image 2 (8x8 = 64 dimensions each)")
        
        return self.descriptors1, self.descriptors2

    def match_features(self, nndr_threshold=0.5, metric='ssd'):
        """
        Step 4: Feature Matching using NNDR (Nearest Neighbor Distance Ratio)
        
        Args:
            nndr_threshold: Threshold for nearest neighbor distance ratio
            metric: Distance metric ('ssd', 'ncc', or 'l2')
        
        Returns:
            matches: List of (idx1, idx2, distance, nndr) tuples
        """
        n1 = len(self.descriptors1)
        n2 = len(self.descriptors2)
        
        matches = []
        nndr_values = []
        
        for i in range(n1):
            desc1 = self.descriptors1[i]
            
            distances = []
            for j in range(n2):
                desc2 = self.descriptors2[j]
                
                if metric == 'ssd':
                  
                    dist = np.sum((desc1 - desc2) ** 2)
                elif metric == 'l2':
                
                    dist = np.linalg.norm(desc1 - desc2)
                elif metric == 'ncc':
                  
                    dist = -np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                distances.append((j, dist))
            
        
            distances.sort(key=lambda x: x[1])
            
          
            nn1_idx, nn1_dist = distances[0]
            nn2_idx, nn2_dist = distances[1]
            
        
            nndr = nn1_dist / (nn2_dist + 1e-7)
            nndr_values.append(nndr)
            
         
            if nndr < nndr_threshold:
                matches.append((i, nn1_idx, nn1_dist, nndr))
        
        self.matches = matches
        self.nndr_values = np.array(nndr_values)
        
        print(f"\nMatching with NNDR threshold = {nndr_threshold}, metric = {metric}")
        print(f"Found {len(matches)} matches out of {n1} features")
        
        return matches


    def visualize_nms(self, output_path='corners_nms.png'):
        """Visualize corners after NMS"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Image 1
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img1_rgb)
        axes[0].plot(self.corners1_nms[:, 1], self.corners1_nms[:, 0], 'r.', markersize=3)
        axes[0].set_title(f'Image 1 - After NMS ({len(self.corners1_nms)} points)')
        axes[0].axis('off')
        
        # Image 2
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        axes[1].imshow(img2_rgb)
        axes[1].plot(self.corners2_nms[:, 1], self.corners2_nms[:, 0], 'r.', markersize=3)
        axes[1].set_title(f'Image 2 - After NMS ({len(self.corners2_nms)} points)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def visualize_nndr_histogram(self, threshold, output_path='nndr_histogram.png'):
        """Visualize NNDR distribution with threshold"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.nndr_values, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold}')
        plt.xlabel('NNDR (Nearest Neighbor Distance Ratio)')
        plt.ylabel('Frequency')
        plt.title('NNDR Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def visualize_top_matches(self, n=5, output_path='top_matches.png'):
        """Visualize top N feature matches showing 1NN and 2NN"""
        if len(self.matches) == 0:
            print("No matches to visualize")
            return
        
        # Sort matches by NNDR (best matches first)
        sorted_matches = sorted(self.matches, key=lambda x: x[3])
        top_matches = sorted_matches[:min(n, len(sorted_matches))]
        
        # Create figure
        fig = plt.figure(figsize=(15, 3*len(top_matches)))
        
        for idx, (i, j, dist, nndr) in enumerate(top_matches):
            # Get patch from image 1
            y1, x1 = self.corners1_nms[i]
            
            # Get 1NN and 2NN from image 2
            # Recompute to get 2NN
            desc1 = self.descriptors1[i]
            distances = []
            for k in range(len(self.descriptors2)):
                desc2 = self.descriptors2[k]
                dist_k = np.sum((desc1 - desc2) ** 2)
                distances.append((k, dist_k))
            distances.sort(key=lambda x: x[1])
            
            nn1_idx = distances[0][0]
            nn2_idx = distances[1][0]
            
            y_nn1, x_nn1 = self.corners2_nms[nn1_idx]
            y_nn2, x_nn2 = self.corners2_nms[nn2_idx]
            
            # Extract patches
            patch_size = 40
            hp = patch_size // 2
            
            patch1 = self.gray1[max(0, y1-hp):y1+hp, max(0, x1-hp):x1+hp]
            patch_nn1 = self.gray2[max(0, y_nn1-hp):y_nn1+hp, max(0, x_nn1-hp):x_nn1+hp]
            patch_nn2 = self.gray2[max(0, y_nn2-hp):y_nn2+hp, max(0, x_nn2-hp):x_nn2+hp]
            
            # Plot
            ax1 = plt.subplot(len(top_matches), 3, idx*3 + 1)
            ax1.imshow(patch1, cmap='gray')
            ax1.set_title(f'Image 1 Feature #{i+1}')
            ax1.axis('off')
            
            ax2 = plt.subplot(len(top_matches), 3, idx*3 + 2)
            ax2.imshow(patch_nn1, cmap='gray')
            ax2.set_title(f'Image 2 NN (1st)\nNNDR={nndr:.3f}')
            ax2.axis('off')
            
            ax3 = plt.subplot(len(top_matches), 3, idx*3 + 3)
            ax3.imshow(patch_nn2, cmap='gray')
            ax3.set_title(f'Image 2 NN (2nd)')
            ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def visualize_matches_lines(self, output_path='matches_visualization.png'):
        """Visualize matches with green lines between matched features"""
        if len(self.matches) == 0:
            print("No matches to visualize")
            return
        
        # Create side-by-side visualization
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show images
        ax1.imshow(img1_rgb)
        ax1.set_title(f'Image 1 (After Feature Matching)\nMatched: {len(self.matches)} | Unmatched: {len(self.corners1_nms) - len(self.matches)} | Total: {len(self.corners1_nms)}')
        ax1.axis('off')
        
        ax2.imshow(img2_rgb)
        ax2.set_title(f'Image 2 (After Feature Matching)\nMatched: {len(self.matches)} | Unmatched: {len(self.corners2_nms) - len(self.matches)} | Total: {len(self.corners2_nms)}')
        ax2.axis('off')
        
        # Plot matched features in green
        matched_idx1 = set()
        matched_idx2 = set()
        
        for i, j, dist, nndr in self.matches:
            y1, x1 = self.corners1_nms[i]
            y2, x2 = self.corners2_nms[j]
            
            matched_idx1.add(i)
            matched_idx2.add(j)
            
            # Draw green lines between matches
            con = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                 axesA=ax1, axesB=ax2, color="green", linewidth=0.5, alpha=0.6)
            ax2.add_artist(con)
            
            # Mark matched points
            ax1.plot(x1, y1, 'go', markersize=3)
            ax2.plot(x2, y2, 'go', markersize=3)
        
        # Plot unmatched features in red
        for i in range(len(self.corners1_nms)):
            if i not in matched_idx1:
                y, x = self.corners1_nms[i]
                ax1.plot(x, y, 'ro', markersize=2, alpha=0.5)
        
        for j in range(len(self.corners2_nms)):
            if j not in matched_idx2:
                y, x = self.corners2_nms[j]
                ax2.plot(x, y, 'ro', markersize=2, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def visualize_matches_colored(self, output_path='matches_colored.png'):
        """Visualize matches with color-coded matched features"""
        if len(self.matches) == 0:
            print("No matches to visualize")
            return
        
        # Create side-by-side visualization
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show images
        ax1.imshow(img1_rgb)
        ax1.set_title(f'Image 1 - Color-coded Matches\n{len(self.matches)} matches')
        ax1.axis('off')
        
        ax2.imshow(img2_rgb)
        ax2.set_title(f'Image 2 - Color-coded Matches\n{len(self.matches)} matches')
        ax2.axis('off')
        
        # Generate colors for each match
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.matches)))
        
        # Plot matched features with same color
        for match_idx, (i, j, dist, nndr) in enumerate(self.matches):
            y1, x1 = self.corners1_nms[i]
            y2, x2 = self.corners2_nms[j]
            
            color = colors[match_idx]
            
            # Plot with matching colors and labels
            ax1.plot(x1, y1, 'o', color=color, markersize=5)
            ax1.text(x1+5, y1+5, str(match_idx+1), color='white', 
                    fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
            
            ax2.plot(x2, y2, 'o', color=color, markersize=5)
            ax2.text(x2+5, y2+5, str(match_idx+1), color='white', 
                    fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")