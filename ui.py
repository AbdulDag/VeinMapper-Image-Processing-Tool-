import cv2
import os
import numpy as np

# --- 1. CONFIGURATION & SETUP ---
dataset_path = r"C:\Users\dagab\Desktop\vein-mapper\Finger Vein Database\002\left"

# Window Setup for Fullscreen
window_name = "VeinMapper Dashboard"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load Images
images = []
try:
    file_list = os.listdir(dataset_path)
    images = [f for f in file_list if f.endswith(('.bmp', '.png', '.jpg'))]
    print(f"System Ready. Loaded {len(images)} patient files.")
except FileNotFoundError:
    print("CRITICAL ERROR: Dataset path not found. Please check your path.")
    exit()

# --- 2. HELPER FUNCTIONS ---

def filter_points(points, min_dist=15):
    """Removes dots that are too close to each other to reduce clutter."""
    clean_points = []
    for p in points:
        is_clutter = False
        for saved_p in clean_points:
            dist = np.sqrt((p[0]-saved_p[0])**2 + (p[1]-saved_p[1])**2)
            if dist < min_dist:
                is_clutter = True
                break
        if not is_clutter:
            clean_points.append(p)
    return clean_points

def create_dashboard(original, enhanced, binary, skeleton, final_map, filename):
    """Builds the futuristic Medical Command Center UI with a gradient background."""
    # Scaling Factors
    MAIN_SCALE = 2.0
    THUMB_SCALE = 0.8
    
    # Prepare Images
    h, w = original.shape[:2]
    main_w, main_h = int(w * MAIN_SCALE), int(h * MAIN_SCALE)
    thumb_w, thumb_h = int(w * THUMB_SCALE), int(h * THUMB_SCALE)
    
    img_large = cv2.resize(original, (main_w, main_h), interpolation=cv2.INTER_CUBIC)
    final_large = cv2.resize(final_map, (main_w, main_h), interpolation=cv2.INTER_NEAREST)
    img_large_color = cv2.cvtColor(img_large, cv2.COLOR_GRAY2BGR)

    thumb_enhanced = cv2.cvtColor(cv2.resize(enhanced, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    thumb_binary = cv2.cvtColor(cv2.resize(binary, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    thumb_skeleton = cv2.cvtColor(cv2.resize(skeleton, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    # --- Create Gradient Background ---
    canvas_w = (main_w * 2) + 40
    canvas_h = 80 + main_h + 40 + thumb_h + 40
    
    # Define Top Color (Professional Blue-Grey) and Bottom Color (Dark Grey) in BGR
    top_color = np.array([70, 50, 50]) 
    bot_color = np.array([25, 25, 25])
    
    # Generate vertical gradient ramp using NumPy linear space
    ramp = np.linspace(top_color, bot_color, canvas_h).astype(np.uint8)
    # Tile it across the width to create the full image
    dashboard = np.tile(ramp[:, None, :], (1, canvas_w, 1))

    # --- Place Elements ---
    # Main Images
    dashboard[80:80+main_h, 20:20+main_w] = img_large_color
    dashboard[80:80+main_h, 20+main_w+10:20+main_w+10+main_w] = final_large

    # Thumbnails
    y_thumb = 80 + main_h + 20
    gap = 20
    start_x = (canvas_w - (thumb_w * 3 + gap * 2)) // 2 
    
    dashboard[y_thumb:y_thumb+thumb_h, start_x:start_x+thumb_w] = thumb_enhanced
    dashboard[y_thumb:y_thumb+thumb_h, start_x+thumb_w+gap:start_x+thumb_w*2+gap] = thumb_binary
    dashboard[y_thumb:y_thumb+thumb_h, start_x+thumb_w*2+gap*2:start_x+thumb_w*3+gap*2] = thumb_skeleton

    # --- Add Text UI with Better Fonts ---
    # Main Title (Thick, bold font)
    cv2.putText(dashboard, "VEIN MAPPER PRO - DIAGNOSTICS MODE", (20, 45), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
    # Subtitle (Cleaner font)
    cv2.putText(dashboard, f"Patient Data ID: {filename}", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.6, (220, 220, 220), 1)
    
    # Thumbnail Labels
    font_s = 0.5
    cv2.putText(dashboard, "Stage 1: Contrast Enhancement", (start_x, y_thumb - 8), cv2.FONT_HERSHEY_DUPLEX, font_s, (255, 255, 100), 1)
    cv2.putText(dashboard, "Stage 2: Vein Extraction", (start_x+thumb_w+gap, y_thumb - 8), cv2.FONT_HERSHEY_DUPLEX, font_s, (255, 255, 100), 1)
    cv2.putText(dashboard, "Stage 3: Skeletonization", (start_x+thumb_w*2+gap*2, y_thumb - 8), cv2.FONT_HERSHEY_DUPLEX, font_s, (255, 255, 100), 1)

    # --- Add Instructions ---
    instruction_text = "[SPACE]: Next Image  |  [Q]: Quit Application"
    # Calculate position for bottom right
    text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
    text_x = canvas_w - text_size[0] - 30
    text_y = canvas_h - 20
    cv2.putText(dashboard, instruction_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 255, 100), 1)

    return dashboard

# --- 3. MAIN PROCESSING LOOP ---

for image_file in images: 
    full_path = os.path.join(dataset_path, image_file)
    frame = cv2.imread(full_path)
    if frame is None: continue 

    # A. CROP
    height, width = frame.shape[:2]
    frame = frame[60:height-60, :]
    
    # B. PRE-PROCESSING
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    veins = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

    # C. MASKING
    _, binary_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=3)
    
    clean_veins = cv2.bitwise_and(veins, veins, mask=mask)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    clean_veins = cv2.morphologyEx(clean_veins, cv2.MORPH_CLOSE, closing_kernel)

    # D. SKELETONIZATION
    skeleton = np.zeros(clean_veins.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = clean_veins.copy()
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0: break

    # E. FEATURE EXTRACTION
    rgb_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    skeleton_bool = skeleton // 255
    rows, cols = np.nonzero(skeleton_bool)
    bifurcations = []
    endpoints = []
    for r, c in zip(rows, cols):
        if r == 0 or r == skeleton.shape[0]-1 or c == 0 or c == skeleton.shape[1]-1: continue
        window = skeleton_bool[r-1:r+2, c-1:c+2]
        neighbors = np.sum(window) - 1
        if neighbors == 1: endpoints.append((c,r))
        elif neighbors >= 3: bifurcations.append((c,r))

    # F. FILTERING & DRAWING
    clean_reds = filter_points(bifurcations, min_dist=20)
    clean_blues = filter_points(endpoints, min_dist=20)
    for pt in clean_reds: cv2.circle(rgb_skeleton, pt, 3, (0, 0, 255), -1)
    for pt in clean_blues: cv2.circle(rgb_skeleton, pt, 3, (255, 0, 0), -1)

    # --- 4. DISPLAY DASHBOARD ---
    final_ui = create_dashboard(gray, enhanced, clean_veins, skeleton, rgb_skeleton, image_file)
    cv2.imshow(window_name, final_ui)
    
    # Controls: Wait for SPACE or Q
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'): break
    # Note: Any other key (like SPACE) will just continue the loop

cv2.destroyAllWindows()