from feature_matching import FeatureMatcher
import os
import sys

def process_image_pair(img1_path, img2_path, output_dir, pair_name):
    """
    Process a single pair of images through the complete pipeline
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_dir: Directory to save outputs
        pair_name: Name prefix for this pair (e.g., 'scene1')
    """
    print(f"\n{'='*80}")
    print(f"Processing {pair_name}")
    print(f"{'='*80}\n")
    

    pair_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    

    matcher = FeatureMatcher(img1_path, img2_path)
    
    # Step 1: Harris Corner Detection
    print("\n--- Step 1: Harris Corner Detection ---")
    matcher.harris_corner_detection()
    matcher.visualize_corners(
        output_path=os.path.join(pair_dir, '1_corners_detected.png')
    )

    window_size = 15  # You can adjust this
    matcher.non_maximal_suppression(window_size=window_size)
    matcher.visualize_nms(
        output_path=os.path.join(pair_dir, '2_corners_nms.png')
    )

    matcher.extract_feature_descriptors(patch_size=40)


    nndr_threshold = 0.5  # You can adjust this
    metric = 'ssd'  # Options: 'ssd', 'l2', 'ncc'
    
    matcher.match_features(nndr_threshold=nndr_threshold, metric=metric)

    print("\n--- Generating Visualizations ---")
    matcher.visualize_nndr_histogram(
        threshold=nndr_threshold,
        output_path=os.path.join(pair_dir, '3_nndr_histogram.png')
    )
    
    matcher.visualize_top_matches(
        n=5,
        output_path=os.path.join(pair_dir, '4_top_matches.png')
    )
    
    matcher.visualize_matches_lines(
        output_path=os.path.join(pair_dir, '5_matches_lines.png')
    )
    
    matcher.visualize_matches_colored(
        output_path=os.path.join(pair_dir, '6_matches_colored.png')
    )
    
    print(f"\n✓ Completed {pair_name}")
    print(f"  Results saved to: {pair_dir}")
    
    return {
        'pair_name': pair_name,
        'num_harris_1': len(matcher.corners1),
        'num_harris_2': len(matcher.corners2),
        'num_nms_1': len(matcher.corners1_nms),
        'num_nms_2': len(matcher.corners2_nms),
        'window_size': window_size,
        'num_matches': len(matcher.matches),
        'nndr_threshold': nndr_threshold,
        'metric': metric,
        'output_dir': pair_dir
    }



def main():

    image1_path = "images/test1.jpeg"
    image2_path = "images/test2.jpeg" 
    

    scene2_img1  = "images/test_two_one.jpeg"
    scene2_img2 = "images/test_two_two.jpeg"


    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(image1_path):
        print(f"Error: Image 1 path '{image1_path}' does not exist.")
        return
    if not os.path.exists(image2_path):
        print(f"Error: Image 2 path '{image2_path}' does not exist.")
        return
    
    results = []
    
    results.append(process_image_pair(image1_path, image2_path, output_dir, "scene1"))
    results.append(process_image_pair(scene2_img1, scene2_img2, output_dir, "scene2"))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for res in results:
        print(f"\n{res['pair_name']}:")
        print(f"  Harris corners: {res['num_harris_1']}, {res['num_harris_2']}")
        print(f"  After NMS (window={res['window_size']}): {res['num_nms_1']}, {res['num_nms_2']}")
        print(f"  Matches (NNDR<{res['nndr_threshold']}, {res['metric']}): {res['num_matches']}")
        print(f"  Output: {res['output_dir']}/")
    
if __name__ == "__main__":
    main()