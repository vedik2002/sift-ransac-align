"""
Generate HTML webpage for homework submission
"""

import os


def generate_html(results_dir='output'):
    """Generate index.html showcasing the results"""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Homework 2 - Automatic Feature Matching</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            color: #555;
            margin-top: 25px;
        }
        
        h4 {
            color: #666;
            margin-top: 20px;
            font-size: 16px;
        }
        
        .scene {
            background: white;
            padding: 30px;
            margin: 30px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .step {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .deliverable {
            margin: 25px 0;
            padding: 20px;
            background: #fff;
            border: 2px solid #3498db;
            border-radius: 5px;
        }
        
        .deliverable-header {
            color: #2980b9;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 15px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 3px;
        }
        
        .step h3 {
            color: #2980b9;
            margin-top: 0;
        }
        
        .images {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }
        
        .image-container {
            flex: 1;
            min-width: 300px;
        }
        
        .image-container img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .image-container p {
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-size: 14px;
        }
        
        .full-width-image {
            width: 100%;
            margin: 20px 0;
        }
        
        .full-width-image img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .stats {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        
        .stats ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .stats li {
            margin: 5px 0;
        }
        
        .info-box {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        .pipeline-diagram {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border: 2px solid #3498db;
            border-radius: 8px;
            text-align: center;
        }
        
        .pipeline-step {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
        }
        
        .arrow {
            display: inline-block;
            color: #3498db;
            font-size: 24px;
            margin: 0 10px;
        }
        
        .requirement {
            background: #f0f8ff;
            border-left: 4px solid #2196F3;
            padding: 10px 15px;
            margin: 10px 0;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>Homework 2: Automatic Feature Matching Across Images</h1>
    
    <div class="info-box">
        <strong>Course:</strong> COMS4732W Computer Vision 2<br>
        <strong>Assignment:</strong> Automatic Feature Matching using Harris Corners and NNDR<br>
        <strong>Implementation:</strong> Python with OpenCV, NumPy, scikit-image, and Matplotlib
    </div>
    
    <h2>Pipeline Overview</h2>
    <div class="pipeline-diagram">
        <div class="pipeline-step">Step 1: Harris Corners</div>
        <span class="arrow">-&gt;</span>
        <div class="pipeline-step">Step 2: NMS</div>
        <span class="arrow">-&gt;</span>
        <div class="pipeline-step">Step 3: Descriptors</div>
        <span class="arrow">-&gt;</span>
        <div class="pipeline-step">Step 4: Matching</div>
    </div>
    
    <h2>Methodology</h2>
    <div class="step">
        <p><strong>Step 1 - Harris Corner Detection:</strong> Detected interest points using the Harris corner detector.</p>
        <p><strong>Step 2 - Non-Maximal Suppression (NMS):</strong> Applied NMS to select only local maxima, reducing redundant corners.</p>
        <p><strong>Step 3 - Feature Descriptors:</strong> Extracted 40x40 axis-aligned patches, downsampled to 8x8 with anti-aliasing, and applied bias/gain normalization to create 64-dimensional descriptors.</p>
        <p><strong>Step 4 - Feature Matching:</strong> Used Nearest Neighbor Distance Ratio (NNDR) with Sum of Squared Differences (SSD) metric to match features.</p>
    </div>
"""
    
    # Add scenes
    scenes = []
    for scene_name in ['scene1', 'scene2']:
        scene_dir = os.path.join(results_dir, scene_name)
        if os.path.exists(scene_dir):
            scenes.append(scene_name)
    
    for scene_name in scenes:
        html_content += f"""
    <div class="scene">
        <h2>{scene_name.upper()}</h2>
        
        <div class="step">
            <h3>Step 1: Harris Corner Detection</h3>
            <p>Detected interest points using Harris corner detector. Red dots indicate detected corners.</p>
            <div class="full-width-image">
                <img src="{scene_name}/1_corners_detected.png" alt="Harris Corners">
            </div>
        </div>
        
        <div class="step">
            <h3>Step 2: Non-Maximal Suppression (NMS)</h3>
            <p>Applied NMS to keep only local maxima, reducing redundant detections.</p>
            <div class="full-width-image">
                <img src="{scene_name}/2_corners_nms.png" alt="After NMS">
            </div>
        </div>
        
        <h3>Step 3 & 4: Feature Matching - Deliverables</h3>
        
        <div class="deliverable">
            <div class="deliverable-header">Deliverable 1: NNDR Histogram with Threshold and Similarity Metric</div>
            <div class="requirement">
                <strong>Requirement:</strong> Display the NNDR histogram and highlight the threshold you used. 
                Also specify which similarity metric you used (e.g. SSD, NCC, etc.).
            </div>
            <p>The histogram shows the distribution of Nearest Neighbor Distance Ratios (NNDR) for all features. 
            The red dashed line indicates the threshold used for filtering matches. Features with NNDR below the threshold 
            are accepted as valid matches (they have one clearly better match). Features with NNDR above the threshold 
            are rejected as ambiguous (multiple similar matches).</p>
            <p><strong>Similarity Metric Used:</strong> SSD (Sum of Squared Differences)</p>
            <p><strong>NNDR Threshold:</strong> 0.5</p>
            <div class="full-width-image">
                <img src="{scene_name}/3_nndr_histogram.png" alt="NNDR Histogram">
            </div>
        </div>
        
        <div class="deliverable">
            <div class="deliverable-header">Deliverable 2: Top 5 Best Feature Matches (1NN and 2NN)</div>
            <div class="requirement">
                <strong>Requirement:</strong> Visualize the 5 best feature matches between the 2 images.<br>
                - First column: feature descriptor for img1's feature<br>
                - Second column: 1NN (first nearest neighbor) feature descriptor from img2<br>
                - Third column: 2NN (second nearest neighbor) feature descriptor from img2
            </div>
            <p>This visualization shows the top 5 matches with the lowest NNDR values (best matches). 
            For each match, we display the feature patch from Image 1, its closest match in Image 2 (1st nearest neighbor), 
            and the second-closest match in Image 2 (2nd nearest neighbor). The NNDR value shown indicates how much better 
            the 1st match is compared to the 2nd match - lower values mean more distinctive matches.</p>
            <div class="full-width-image">
                <img src="{scene_name}/4_top_matches.png" alt="Top 5 Matches">
            </div>
        </div>
        
        <div class="deliverable">
            <div class="deliverable-header">Deliverable 3: Match Visualization</div>
            <div class="requirement">
                <strong>Requirement:</strong> Visualize the matches using one of the 2 options:<br>
                - Option 1: Color-code the matched features across both images and display them side-by-side. 
                  Also, put a number next to each feature to indicate the match index.<br>
                - Option 2: Draw green lines between matched features across both images side-by-side, 
                  and put red dots on the unmatched features.
            </div>
            
            <h4>Option 1: Color-Coded Matches</h4>
            <p>Each matched feature pair is shown with the same color across both images. 
            Numbers indicate the match index, making it easy to identify corresponding features.</p>
            <div class="full-width-image">
                <img src="{scene_name}/6_matches_colored.png" alt="Color-coded Matches">
            </div>
            
            <h4>Option 2: Line Connections with Matched/Unmatched Features</h4>
            <p><strong>Green dots and lines:</strong> Matched features between the two images.<br>
            <strong>Red dots:</strong> Unmatched features (rejected by NNDR threshold as ambiguous).</p>
            <div class="full-width-image">
                <img src="{scene_name}/5_matches_lines.png" alt="Line Matches">
            </div>
        </div>
    </div>
"""
    
    html_content += """
    <h2>Discussion</h2>
    <div class="step">
        <h3>Parameters Used</h3>
        <ul>
            <li><strong>Harris Corner Detection:</strong> sigma=1.5, threshold=0.10 (10% of max response)</li>
            <li><strong>NMS Window Size:</strong> 20x20 pixels</li>
            <li><strong>Feature Descriptor Patch Size:</strong> 40x40 pixels (downsampled to 8x8 = 64 dimensions)</li>
            <li><strong>Anti-aliasing:</strong> Enabled (Gaussian blur before downsampling)</li>
            <li><strong>NNDR Threshold:</strong> 0.5</li>
            <li><strong>Distance Metric:</strong> SSD (Sum of Squared Differences)</li>
        </ul>
        
        <h3>Observations</h3>
        <p>The feature matching pipeline successfully identified corresponding points between image pairs. 
        The NNDR threshold of 0.5 effectively filtered out ambiguous matches, keeping only distinctive features. 
        The histogram shows a clear separation between good matches (low NNDR, left side) and poor/ambiguous matches 
        (high NNDR, right side).</p>
        
        <p>The Harris corner detector found many interest points, particularly at building edges, window corners, 
        and architectural details. NMS successfully reduced redundancy while preserving the most salient corners. 
        The 8x8 downsampled patch descriptors with anti-aliasing provided a good balance between distinctiveness 
        and computational efficiency.</p>
        
        <p>The top 5 matches visualization clearly shows that the 1st nearest neighbors are visually similar to 
        the query features, while the 2nd nearest neighbors are significantly different, validating the effectiveness 
        of the NNDR approach for rejecting ambiguous matches.</p>
    </div>
    
    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #888;">
        <p>COMS4732W Computer Vision 2 - Homework 2</p>
    </footer>
</body>
</html>
"""
    
    # Write HTML file with UTF-8 encoding
    output_path = os.path.join(results_dir, 'index.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated webpage: {output_path}")
    print(f"Open it in a browser to view your results!")


if __name__ == "__main__":
    generate_html()