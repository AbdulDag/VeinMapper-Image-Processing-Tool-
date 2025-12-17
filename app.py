from flask import Flask, render_template, Response, jsonify
import cv2
import os
import numpy as np
import random

app = Flask(__name__)

# --- CONFIGURATION ---
dataset_path = r"C:\Users\dagab\Desktop\vein-mapper\Finger Vein Database\002\left"
current_index = 0 # Tracks which image in the folder is currently being processed

images = []
try:
    # List all files in the directory
    file_list = os.listdir(dataset_path) 
    images = [f for f in file_list if f.endswith(('.bmp', '.png', '.jpg'))] # Filter list to keep only image files (bmp, png, jpg)
    print(f"Server Ready. Loaded {len(images)} patient files.")
except FileNotFoundError:
    # Safety check in case the folder path is typed incorrectly or missing
    print("CRITICAL ERROR: Dataset path not found.")

# --- HELPER: Draw Text with Background ---
def draw_label(img, text):
    """Draws white text with a black background box for readability"""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thickness = 1
    margin = 5
    
    # 1. Get text sizes, how much space texts needs
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # 2. Draw black rectangle (Background)
    # Top-left corner is (0,0), Bottom-right is (text_width + margin, text_height + margin * 2)
    cv2.rectangle(img, (0, 0), (text_width + margin * 2, text_height + margin * 3), (0, 0, 0), -1)
    
    # 3. Draw actual white text on top of the black box
    cv2.putText(img, text, (margin, text_height + margin), font, scale, (255, 255, 255), thickness)

# --- PROCESSING PIPELINE ---
def create_tiled_dashboard(enhanced, binary, skeleton, final_map):
    target_w, target_h = 400, 300
    
    # Preparing Contrast Enhanced Image (Top-Left)
    img1 = cv2.resize(enhanced, (target_w, target_h))
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) # Convert to color so it can be combined with others
    draw_label(img1, "1. Contrast Enhanced")

    # 2. Binary Extraction (Thick Veins)
    img2 = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    draw_label(img2, "2. Binary Extraction")

    # 3. Skeleton (Thin Lines)
    skeleton_disp = cv2.resize(skeleton, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    # Dilate (thicken) the lines slightly so the 1-pixel thin skeleton is visible on screen
    kernel = np.ones((2,2), np.uint8)
    kernel = np.ones((2,2), np.uint8) 
    skeleton_disp = cv2.dilate(skeleton_disp, kernel, iterations=1)
    img3 = cv2.cvtColor(skeleton_disp, cv2.COLOR_GRAY2BGR)
    draw_label(img3, "3. Skeleton Structure")

    # Prepare Final Biometric Map (Bottom-Right)
    img4 = cv2.resize(final_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    draw_label(img4, "4. Biometric Map")

    # Combine
    top_row = np.hstack((img1, img2))
    bot_row = np.hstack((img3, img4))
    grid = np.vstack((top_row, bot_row))
    
    return grid

def get_processed_frame():
    global current_index
    if not images: return np.zeros((600,800,3), np.uint8) # Return black screen if no images
    
    # Load the current image from the list
    full_path = os.path.join(dataset_path, images[current_index])
    frame = cv2.imread(full_path)
    if frame is None: return np.zeros((600,800,3), np.uint8)

    # A. CROP & CONTRAST
    # Crop 60 pixels from top and bottom to remove potential sensor noise/borders
    h, w = frame.shape[:2]
    frame = frame[60:h-60, :] 

    # Convert to Grayscale and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This makes the dark veins stand out much more clearly against the skin
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # B. THRESHOLDING
    # Use Adaptive Thresholding to separate veins (dark) from skin (light)
    veins = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    
    # Cleanup Mask
    _, binary_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    # Find the largest object (the finger) and fill it with white
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=3)
    # Apply the mask: keep veins only where the mask is white
    clean_veins = cv2.bitwise_and(veins, veins, mask=mask)
    # Use 'closing' (dilation then erosion) to fill small gaps inside the veins
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean_veins = cv2.morphologyEx(clean_veins, cv2.MORPH_CLOSE, closing_kernel)

    # C. SKELETONIZATION (Robust)
    skeleton = np.zeros(clean_veins.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = clean_veins.copy()
    
    # Loop that eats away at the edges of the veins
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img = eroded.copy()
        # Stop when there is nothing left to erode
        if cv2.countNonZero(temp_img) == 0: break

    # D. FEATURE DOTS
    rgb_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    skeleton_bool = skeleton // 255 # Convert 0-255 range to 0-1 for easier counting
    rows, cols = np.nonzero(skeleton_bool)
    bifurcations = [] 
    endpoints = []
    
    for r, c in zip(rows, cols):
        # Skip pixels on the very edge of the image frame
        if r == 0 or r == skeleton.shape[0]-1 or c == 0 or c == skeleton.shape[1]-1: continue
        window = skeleton_bool[r-1:r+2, c-1:c+2]
        # Count how many neighbors the pixel has
        neighbors = np.sum(window) - 1
        if neighbors == 1: endpoints.append((c,r))
        elif neighbors >= 3: bifurcations.append((c,r))

    def filter_points(points, min_dist=15):
        clean_points = []
        for p in points:
            is_clutter = False
            for saved_p in clean_points:
                # Calculate Euclidean distance between points
                dist = np.sqrt((p[0]-saved_p[0])**2 + (p[1]-saved_p[1])**2)
                if dist < min_dist:
                    is_clutter = True; break
            if not is_clutter: clean_points.append(p)
        return clean_points
    
    # Filter the lists using a 20-pixel minimum distance
    clean_reds = filter_points(bifurcations, min_dist=20)
    clean_blues = filter_points(endpoints, min_dist=20)
    
    # Draw Red circles for splits and Blue circles for ends
    for pt in clean_reds: cv2.circle(rgb_skeleton, pt, 4, (0, 0, 255), -1)
    for pt in clean_blues: cv2.circle(rgb_skeleton, pt, 4, (255, 0, 0), -1)

    # E. SEND TO DASHBOARD
    return create_tiled_dashboard(enhanced, clean_veins, skeleton, rgb_skeleton)

# --- ROUTES ---
@app.route('/')
# Serves the main HTML page
def index(): return render_template('index.html')

@app.route('/dashboard_image')
def dashboard_image():
    # Generates the processed 4-pane dashboard and streams it as a JPEG image
    frame = get_processed_frame()
    ret, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    # Sends stats (total images, quality, patient ID) to the dashboard's sidebar/UI
    return jsonify({
        "cards": {"total": len(images), "today": 24, "quality": "92%", "pending": 3},
        "chart_daily": {"labels":["M","T","W","T","F","S","S"], "data":[30,40,35,50,49,60,70]},
        "chart_quality": {"labels":["Good","Fair","Poor"], "data":[70,20,10]},
        "recent_activity": [
            {"time":"10:00","event":"Scan Verified","status":"Done"},
            {"time":"09:45","event":"Dr. Login","status":"Active"}
        ],
        "current_patient_id": images[current_index] if images else "N/A"
    })

@app.route('/next', methods=['POST'])
def next_image():
    # Moves to the next image in the folder (loops back to start if at the end)
    global current_index
    if images: current_index = (current_index + 1) % len(images)
    return jsonify(success=True)

@app.route('/prev', methods=['POST'])
def prev_image():
    # Moves to the previous image in the folder
    global current_index
    if images: current_index = (current_index - 1) % len(images)
    return jsonify(success=True)

if __name__ == '__main__':
    # Starts the web server on port 5000
    app.run(debug=True, port=5000)