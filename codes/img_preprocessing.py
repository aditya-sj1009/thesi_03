import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import color
from sklearn.model_selection import train_test_split

def white_balance(img):
    """Applies a simple white balance using the gray world assumption."""
    avg_color_per_row = np.average(img, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
    avg_b, avg_g, avg_r = avg_colors

    balance_b = 128 / avg_b
    balance_g = 128 / avg_g
    balance_r = 128 / avg_r

    balanced_img = img.copy()
    balanced_img[:, :, 0] = np.clip(balanced_img[:, :, 0] * balance_b, 0, 255)
    balanced_img[:, :, 1] = np.clip(balanced_img[:, :, 1] * balance_g, 0, 255)
    balanced_img[:, :, 2] = np.clip(balanced_img[:, :, 2] * balance_r, 0, 255)

    return balanced_img.astype(np.uint8)


def clahe_enhancement(img):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def detect_color_card_updated(image):
    """Detect and extract the jaundice color card from the image with improved robustness"""
    # Check if image exists
    if image is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check for low contrast
    from skimage.exposure import is_low_contrast
    if is_low_contrast(gray, fraction_threshold=0.35):
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    
    # Use Canny edge detection instead of thresholding
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect any broken lines
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours on the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Try multiple approximation parameters if needed
    for epsilon_factor in [0.02, 0.01, 0.04, 0.06]:
        card_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                
                # Accept quadrilaterals (4 vertices)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    card_contour = approx
        
        if card_contour is not None:
            break
    
    if card_contour is None:
        # If still no quadrilateral found, try with largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest_contour, True)
            card_contour = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        else:
            return None
    
    # Rest of your perspective transform code remains the same
    card_contour = card_contour.reshape(-1, 2)
    
    # Sort corners (top-left, top-right, bottom-right, bottom-left)
    # First sort by x to separate left/right points
    sorted_by_x = card_contour[np.argsort(card_contour[:, 0])]
    left_pts = sorted_by_x[:2]
    right_pts = sorted_by_x[2:]
    
    # Sort left points by y (top has smaller y)
    left_pts = left_pts[np.argsort(left_pts[:, 1])]
    # Sort right points by y
    right_pts = right_pts[np.argsort(right_pts[:, 1])]
    
    # Reorder as [tl, tr, br, bl]
    src_pts = np.array([
        left_pts[0],   # top-left
        right_pts[0],  # top-right
        right_pts[1],  # bottom-right
        left_pts[1]    # bottom-left
    ], dtype=np.float32)
    
    # Define destination points for a square output
    card_size = 500  # Output size in pixels
    dst_pts = np.array([
        [0, 0],
        [card_size, 0],
        [card_size, card_size],
        [0, card_size]
    ], dtype=np.float32)
    
    # Get perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    card_image = cv2.warpPerspective(image, M, (card_size, card_size))
    
    return card_image

def detect_color_card_updated_1(image):
    """Improved color card detection for low contrast and blurry images"""
    # Check if image exists
    if image is None:
        return None
        
    # Make a copy to avoid modifying the original
    original = image.copy()
    
    # Step 1: Check for low contrast and enhance if needed
    from skimage.exposure import is_low_contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if is_low_contrast(gray, fraction_threshold=0.25):
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Also enhance original image
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Step 2: Apply bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Step 3: Use adaptive thresholding instead of global
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Step 4: Apply morphological operations to clean up the threshold
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Try Canny edge detection as an alternative
    edges = cv2.Canny(blurred, 30, 200)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 6: Find contours on both threshold and edges
    contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine contours from both methods
    all_contours = contours_thresh + contours_edges
    
    # Step 7: Try multiple epsilon values for approximation
    card_contour = None
    
    for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
        for contour in sorted(all_contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip very small contours
                continue
                
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
            
            # Check if it's approximately a rectangle (4 points)
            if len(approx) == 4:
                card_contour = approx
                break
                
        if card_contour is not None:
            break
    
    # Step 8: If no quadrilateral found, fall back to largest contour
    if card_contour is None:
        if all_contours:
            largest = max(all_contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            card_contour = cv2.approxPolyDP(largest, 0.02 * peri, True)
        else:
            # Last resort: use the entire image
            h, w = image.shape[:2]
            card_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    # Reshape and continue with perspective transform
    card_contour = card_contour.reshape(-1, 2)
    
    # If we don't have exactly 4 points, we need to fix this
    if len(card_contour) != 4:
        # Convert to rectangle using bounding box
        x, y, w, h = cv2.boundingRect(card_contour)
        card_contour = np.array([
            [x, y],
            [x+w, y],
            [x+w, y+h],
            [x, y+h]
        ])
    
    # Sort points as before
    sorted_by_x = card_contour[np.argsort(card_contour[:, 0])]
    left_pts = sorted_by_x[:2]
    right_pts = sorted_by_x[2:]
    
    left_pts = left_pts[np.argsort(left_pts[:, 1])]
    right_pts = right_pts[np.argsort(right_pts[:, 1])]
    
    src_pts = np.array([
        left_pts[0],
        right_pts[0],
        right_pts[1],
        left_pts[1]
    ], dtype=np.float32)
    
    # Define destination points and perform perspective transform
    card_size = 500
    dst_pts = np.array([
        [0, 0],
        [card_size, 0],
        [card_size, card_size],
        [0, card_size]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    card_image = cv2.warpPerspective(original, M, (card_size, card_size))
    
    return card_image


def detect_color_card(image):
    """Detect and extract the jaundice color card from the image"""
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find edges
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find square/rectangular contours (the card)
    card_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small contours
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # If it's roughly a quadrilateral and has significant area
            if len(approx) == 4 and area > max_area:
                max_area = area
                card_contour = approx
    
    if card_contour is None:
        return None
    
    # Perspective transform to get top-down view of card
    card_contour = card_contour.reshape(-1, 2)
    
    # Sort corners (top-left, top-right, bottom-right, bottom-left)
    # First sort by x to separate left/right points
    sorted_by_x = card_contour[np.argsort(card_contour[:, 0])]
    left_pts = sorted_by_x[:2]
    right_pts = sorted_by_x[2:]
    
    # Sort left points by y (top has smaller y)
    left_pts = left_pts[np.argsort(left_pts[:, 1])]
    # Sort right points by y
    right_pts = right_pts[np.argsort(right_pts[:, 1])]
    
    # Reorder as [tl, tr, br, bl]
    src_pts = np.array([
        left_pts[0],   # top-left
        right_pts[0],  # top-right
        right_pts[1],  # bottom-right
        left_pts[1]    # bottom-left
    ], dtype=np.float32)
    
    # Define destination points for a square output
    card_size = 500  # Output size in pixels
    dst_pts = np.array([
        [0, 0],
        [card_size, 0],
        [card_size, card_size],
        [0, card_size]
    ], dtype=np.float32)
    
    # Get perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    card_image = cv2.warpPerspective(image, M, (card_size, card_size))
    
    return card_image

# img_path = 'data/img_dataset/33_PM_1.jpeg'
# img = cv2.imread(img_path)
# card = detect_color_card_updated(img)
# cv2.imwrite('card_33_PM_1.jpeg', card)

def extract_regions(card_image):
    """Extract the skin region and reference color patches from the card"""
    h, w = card_image.shape[:2]
    
    # Define approximate regions based on card layout
    # These will work even if card orientation varies because we've corrected perspective
    
    # Center region (skin patch) - approximately 1/3 of card width/height
    skin_size = min(w, h) // 3
    skin_x1 = (w - skin_size) // 2
    skin_y1 = (h - skin_size) // 2
    skin_x2 = skin_x1 + skin_size
    skin_y2 = skin_y1 + skin_size
    
    skin_patch = card_image[skin_y1:skin_y2, skin_x1:skin_x2]
    
    # Reference color patches - using color-based segmentation
    # Convert to HSV for easier color filtering
    hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for each reference patch
    color_ranges = {
        'yellow': ([20, 100, 100], [40, 255, 255]),
        'cyan': ([85, 50, 50], [110, 255, 255]),
        'magenta': ([140, 50, 50], [170, 255, 255]),
        'beige': ([10, 30, 160], [25, 80, 220])
    }
    
    ref_patches = {}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Find largest contour in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Extract with a small margin to avoid edges
            margin = 5
            ref_patches[color] = card_image[y+margin:y+h-margin, x+margin:x+w-margin]
    
    return skin_patch, ref_patches

def extract_color_features(patch):
    """Extract color features from a skin or reference patch"""
    if patch is None or patch.size == 0:
        return {}
        
    # Convert to different color spaces
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    
    features = {}
    
    # Mean and standard deviation for each channel
    for name, color_space in zip(['rgb', 'hsv', 'lab'], [rgb, hsv, lab]):
        means = np.mean(color_space, axis=(0, 1))
        stds = np.std(color_space, axis=(0, 1))
        
        for i, channel in enumerate(['1', '2', '3']):
            features[f'{name}_mean_{channel}'] = float(means[i])
            features[f'{name}_std_{channel}'] = float(stds[i])
    
    # Calculate yellow-specific features for jaundice detection
    
    # LAB b-channel (yellowing indicator)
    b_channel = lab[:, :, 2]
    features['lab_b_mean'] = float(np.mean(b_channel))
    
    # Yellow hue ratio in HSV
    h_channel = hsv[:, :, 0]
    yellow_hues = ((h_channel >= 20) & (h_channel <= 60)).sum()
    features['yellow_hue_ratio'] = float(yellow_hues) / (patch.shape[0] * patch.shape[1])
    
    return features

def process_jaundice_image(image_path):
    """Process a jaundice card image and extract relevant features"""
    # Read image
    # print(image_path)
    img = cv2.imread(image_path)
    if img is None:
        # print(f"Error: Could not read image {image_path}")
        return {'error': f'Could not read image: {image_path}'}
    
    # Detect and extract color card
    card_image = detect_color_card(img)
    
    # print('card_image created')
    if card_image is None:
        # print(f"Error: Could not detect color card in {image_path}")
        return None, None
    
    # Extract skin and reference patches
    # print('skin')
    skin_patch, ref_patches = extract_regions(card_image)
    
    # Extract color features
    # print('features')
    skin_features = extract_color_features(skin_patch)
    ref_features = {color: extract_color_features(patch) 
                   for color, patch in ref_patches.items()}
    
    # Calculate normalized features (to account for lighting conditions)
    # print('normalize')
    normalized_features = {}
    if 'yellow' in ref_patches:
        yellow_features = ref_features['yellow']
        for key in skin_features:
            if key.startswith('rgb_mean') or key.startswith('lab_'):
                normalized_features[f'norm_{key}'] = skin_features[key] / max(yellow_features.get(key, 1), 1)
    
    # Store extracted patches for visualization/debugging
    # result = np.array(list(normalized_features.values()))
        # 'skin_patch': skin_patch,
        # 'ref_patches': ref_patches,
        # 'skin_features': skin_features,
        # 'ref_features': ref_features,
    # print(f"Extracted features for {image_path}: {normalized_features}")
    # print(f"Skin features: {skin_features}")
    return normalized_features
    
def enhance_image_quality(image):
    """Apply preprocessing to improve image quality before analysis"""
    # White balance correction
    wb = cv2.xphoto.createSimpleWB()
    balanced = wb.balanceWhite(image.copy())
    
    # Contrast enhancement
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_medical_data(medical_records_df):
    """Process medical record data"""
    # Normalize weight
    medical_records_df['weight_normalized'] = (medical_records_df['Weight (g)'] - 
                                              medical_records_df['Weight (g)'].mean()) / medical_records_df['Weight (g)'].std()
    
    # Process postnatal age (might want to bin or normalize)
    medical_records_df['postnatal_age_normalized'] = (medical_records_df['Postnatal Age (hrs)'] - 
                                                     medical_records_df['Postnatal Age (hrs)'].mean()) / medical_records_df['Postnatal Age (hrs)'].std()
    
    # One-hot encode treatment if categorical
    # treatment_encoded = pd.get_dummies(medical_records_df['treatment'], prefix='treatment')
    # medical_records_df = pd.concat([medical_records_df, treatment_encoded], axis=1)
    
    return medical_records_df


def create_combined_features( medical_records_df):
    """Combine image features with medical record data"""
    # Ensure both dataframes have matching patient IDs
    data= []
    for _,row in medical_records_df.iterrows():
        print('processing', row['new_image_name'])
        img = row['new_image_name']
        img_path = os.path.join('data/img_dataset', img)
        img_features = process_jaundice_image(img_path)
        if img_features is None:
            # print(f"Error: No features extracted for image {img}")
            continue
        
        # Handle tuple return type from process_jaundice_image
        if isinstance(img_features, tuple):
            # Unpack tuple assuming structure is (normalized_features, skin_features)
            normalized_features, skin_features = img_features
            
            # Create combined features dictionary
            features_dict = {}
            
            # Add normalized features if available
            if isinstance(normalized_features, dict):
                features_dict.update(normalized_features)
                
            # Add skin features if available
            if isinstance(skin_features, dict):
                features_dict.update(skin_features)
        else:
            # If img_features is already a dictionary
            features_dict = img_features
        
        
        clinical_data = medical_records_df[medical_records_df['new_image_name'] == img]
        if clinical_data.empty:
            print(f"Warning: No clinical data found for image {img}")
            continue
        combined_entry = {
            ** features_dict, 
            ** row.to_dict()
        }
        data.append(combined_entry)
        
    combined_df = pd.DataFrame(data)
    combined_df = combined_df.drop('new_image_name', axis=1)
    clean_df = combined_df.dropna()
    
    return clean_df


# try:
#     clinical_data_df = pd.read_csv('data/clinical_data_updated.csv')
#     clinical_data_df = process_medical_data(clinical_data_df)
#     print("Loaded combined dataset with shape:", clinical_data_df.shape )
# except FileNotFoundError:
#     print("Error: Run preprocessing first to generate combined_features.csv")
#     breakpoint

# combined_df = create_combined_features( clinical_data_df)
# # print(combined_df[4])
# # 2. Prepare data for modeling
# X = combined_df.drop(columns=['Jaundice Decision']).values
# y = combined_df['Jaundice Decision'].values
# feature_names = combined_df.drop(columns=['Jaundice Decision']).columns.tolist()
# print(f"\nData shape: {X.shape}, Labels shape: {y.shape}")
# print( feature_names)
# print (X[0:5])

# # 3. Split data into train/test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, 
#     test_size=0.2, 
#     stratify=y,
#     random_state=42
# )
# np.savez('train_test_split.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
# np.savez('feature_names.npz', feature_names=np.array(feature_names, dtype=object))
# print(f"\nTrain set: {X_train.shape[0]} samples")
# print(f"Test set: {X_test.shape[0]} samples")

def extract_cnn_features(image_path, feature_extractor):
    """Extract CNN features from jaundice color card"""
    # Preprocess image using your existing functions
    img = cv2.imread(image_path)
    card = detect_color_card_updated(img)
    if card is None:
        return None
    
    # Standardize image size for CNN
    card = cv2.resize(card, (128, 128))  # Match CNN input shape
    card = card / 255.0  # Normalize
    
    # Extract features
    features = feature_extractor.predict(card[np.newaxis, ...], verbose=0)
    return features.flatten().astype(np.float32)  # Convert to 1D vector


