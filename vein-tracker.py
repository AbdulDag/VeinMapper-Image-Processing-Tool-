import cv2 #OpenCV for video capture and display
import os
import numpy as np





# -- Dataset Setup -- 
dataset_path = r"C:\Users\dagab\Desktop\vein-mapper\Finger Vein Database\002\left"  # Path to the dataset directory
UI_SCALE = 2.0
try:
    file_list = os.listdir(dataset_path) # Getting a list of files in dataset directory
    images = [f for f in file_list if f.endswith(('.bmp', '.png', '.jpg'))] # filters Image files and only keeps files with vaild image extentsions
    print(f"Found {len(images)} images in the dataset.") # Displays total images found 
    print("First image name:", images[0]) # Display name of first letter sample   
except FileNotFoundError: # Handles case where folder path deos not exists
    print("Error: The folder path is wrong. Check the path in 'dataset_path'")

# -- Processing Each Image in Dataset -- 
# Loops through each image filename in filterd images list 
for image_file in images: 
    full_path = os.path.join(dataset_path, image_file) # Construct complete file path by joining directory path and filename

    frame = cv2.imread(full_path) # Reads image file from disk into memory as numpy array
    if frame is None: # Checking if image failed to load
        print(f"Error loading image: {full_path}")
        continue # Skips to next image in the loop 
    height, width = frame.shape[:2] # Gettins the dimensions of the image (height and width)
    frame = frame[60:height-60, :]
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the image to greyscale
    
    # -- Enhance image contrast --
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Creating a clahe (Contrast Limited Adaptive Histogram Equalization) object 
    enhanced = clahe.apply(gray)

    # Now we add adaptive thresholding
    # We make the veins pure white and then use gaussian blur to smooth the image

    # Change block size and c value respectively to adjust vein visibility. block size is basically for the noise and if vein broken or invisible lower c value. if too much noise increase block size
    veins = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)



    # Add masking cuz its detecting outside the finger area too
    # Tweak second parameter if it cuts off too mcuh of finger or includes too much background
    _, binary_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find all shapes in the image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    # Checking if countor were found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea) # Finds largest contour and sorts by their area
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED) # Draws largest contour on mask as a filled white shape
        mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=3) # Removes noisy egde articats 
    # This keeps the 'veins' ONLY where 'finger_mask' is White.
    clean_veins = cv2.bitwise_and(veins, veins, mask=mask)

    # This removes small white dots (noise)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    clean_veins = cv2.morphologyEx(clean_veins, cv2.MORPH_CLOSE, closing_kernel)

    # ... Now start the SKELETONIZATION loop ...
    skeleton = np.zeros(clean_veins.shape, np.uint8)
    
    # Get a structural element (a cross shape works best for lines)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Loop until no white pixels are left in the temp image
    temp_img = clean_veins.copy()
    #skeleton loop
    while True:
        # Erode (shrink) the white lines
        eroded = cv2.erode(temp_img, element)
        
        # Dilate (grow) them back to see what we lost
        temp = cv2.dilate(eroded, element)
        
        # Subtract to find the "edges" that were removed
        temp = cv2.subtract(temp_img, temp)
        
        # Add those edges to our permanent skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)
        
        # Update for next loop
        temp_img = eroded.copy()
        
        # Stop if the image is completely black
        if cv2.countNonZero(temp_img) == 0:
            break
  
    
    # 1. Prepare color image
    rgb_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR) #Converts greysacle skeleton to BGR color format
    
    # 2. Helper lists to store "candidate" points
    # We collect them first, then filter them
    bifurcations = [] # Red candidates
    endpoints = []    # Blue candidates
    
    # 3. Analyze the structure (Standard Crossing Number)
    skeleton_bool = skeleton // 255
    rows, cols = np.nonzero(skeleton_bool) # Gets coordinates of all white pixels
    
    # Loops through each vein pixel to classify as endpoint or normal point
    for r, c in zip(rows, cols):
        if r == 0 or r == skeleton.shape[0]-1 or c == 0 or c == skeleton.shape[1]-1:
            continue
            
        window = skeleton_bool[r-1:r+2, c-1:c+2]
        neighbors = np.sum(window) - 1 # Counts the number of white neighbors not including center pixels itself
        
        if neighbors == 1: # Classifying point based on number of neighbors ( 1 neighbor = endppint)
            endpoints.append((c,r))      # Add to blue list
        elif neighbors >= 3:
            bifurcations.append((c,r))   # Add to red list

    # --- THE CLUTTER FILTER ---
    # Function to remove dots that are too close to each other
    def filter_points(points, min_dist=15): # Prevents duplicate/clustered detections of same feature
        clean_points = [] # Initializing empty list to store filtered points 
        for p in points: # Loop for each point to check if its too close to already saved points 
            # Check if 'p' is close to any point we already saved
            is_clutter = False
            for saved_p in clean_points: # Calculates Eculidean distance between points 
                dist = np.sqrt((p[0]-saved_p[0])**2 + (p[1]-saved_p[1])**2)
                if dist < min_dist: #If distance is less then minium threshold, mark as clutter 
                    is_clutter = True
                    break
            if not is_clutter: #Only add point if it NOT too close to existing points 
                clean_points.append(p)
        return clean_points

    # 4. Filter the lists
    clean_reds = filter_points(bifurcations, min_dist=20) # Stricter for splits
    clean_blues = filter_points(endpoints, min_dist=20)

    # 5. Draw ONLY the clean points
    for pt in clean_reds:
        cv2.circle(rgb_skeleton, pt, 3, (0, 0, 255), -1) # Red
        
    for pt in clean_blues:
        cv2.circle(rgb_skeleton, pt, 3, (255, 0, 0), -1) # Blue
    # Displays the vein network with bifurcations red and blue endpoints marked 
    cv2.imshow("Vein Network Structure", rgb_skeleton) 
    
    # Show the final result with keypoints
    cv2.imshow("Skeleton (Final Map)", skeleton)
    
    # ... then your other imshow lines ...
    cv2.imshow("vein image of ahmed sial", clean_veins)
    cv2.imshow("raw image of ahmed sial", gray) # Display the original greyscale finger image
    cv2.imshow("enhanced image of ahmed sial", enhanced) # Display the contrast-enhanced version of the image

    
    # Check if 'q' key was pressed to quit program
    key =  cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break # Exit while loop, ending video cpature 
cv2.destroyAllWindows()