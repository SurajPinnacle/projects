import cv2
import numpy as np
import os
import csv

# Function to preprocess the image (gray scaling)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check the file path or file integrity.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# Function to check for overlap between detected icons
def check_overlap(detected_points, new_box, overlap_threshold=0.3):
    x1, y1, x2, y2 = new_box
    for existing_box in detected_points:
        xa1, ya1, xa2, ya2 = existing_box
        # Compute intersection
        inter_x1 = max(x1, xa1)
        inter_y1 = max(y1, ya1)
        inter_x2 = min(x2, xa2)
        inter_y2 = min(y2, ya2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        # Compute union area
        box_area = (x2 - x1) * (y2 - y1)
        existing_area = (xa2 - xa1) * (ya2 - ya1)
        union_area = box_area + existing_area - inter_area
        # Compute IoU (Intersection over Union)
        if union_area == 0:
            continue
        iou = inter_area / union_area
        if iou > overlap_threshold:
            return True
    return False

# Function to perform template matching on an image
def match_template(img_gray, template, threshold):
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    return loc

# Function to load all templates from a folder
def load_templates_from_folder(folder_path):
    template_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            template_paths.append(os.path.join(folder_path, filename))
    return template_paths

# Function to capture detected object coordinates
def capture_detected_coordinates(image_path, template_folder, scales, threshold):
    # Load and preprocess the main image
    img_gray = preprocess_image(image_path)
    
    # List to store detected coordinates (bounding boxes)
    detected_coordinates = []
    
    # Load all templates from the folder
    template_paths = load_templates_from_folder(template_folder)

    # Iterate over each template
    for template_path in template_paths:
        # Load and preprocess the template
        template = cv2.imread(template_path, 0)  # Load as grayscale
        if template is None:
            print(f"Skipping invalid template: {template_path}")
            continue  # Skip invalid templates
        original_w, original_h = template.shape[::-1]

        # Iterate over scales
        for scale in scales:
            # Resize the template according to the scale
            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)
            if scaled_w <= 0 or scaled_h <= 0:
                continue  # Skip invalid sizes
            template_resized = cv2.resize(template, (scaled_w, scaled_h))

            w_rotated, h_rotated = template_resized.shape[::-1]
            loc = match_template(img_gray, template_resized, threshold)

            # Iterate over all matched locations
            for pt in zip(*loc[::-1]):  # Switch columns and rows
                x1, y1 = pt
                x2, y2 = x1 + w_rotated, y1 + h_rotated
                new_box = [x1, y1, x2, y2]

                # Check for overlap with existing detections
                if not check_overlap(detected_coordinates, new_box):
                    detected_coordinates.append(new_box)

    return detected_coordinates

# Function to mark the detected objects on the image and save it
def mark_the_objects(image_path, detected_coordinates):
    # Load the image to mark the detections
    img = cv2.imread(image_path)

    # Draw rectangles on the detected image
    for coord in detected_coordinates:
        x1, y1, x2, y2 = coord
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw rectangle in red color

    # Detected Image
    object_count = len(detected_coordinates)

    return img, object_count

# Function to save the image in the folder
def save_the_image(img, object_count, output_folder, icon_name):
    # Save the detected image
    filename = f"{icon_name}_{object_count}_detections.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, img)

    # Path for the CSV file
    csv_filepath = os.path.join(output_folder, 'detected_objects.csv')

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_filepath)

    # Open the CSV file (create if not exists)
    with open(csv_filepath, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        # If the file doesn't exist, write the header row
        if not file_exists:
            csv_writer.writerow(['icon_name', 'object_count'])

        # Write the new data (icon_name, object_count)
        csv_writer.writerow([icon_name, object_count])

    print(f"Output image saved to: {filepath}")
    print(f"Total Detected icons are: {object_count}")
    print(f"CSV file updated: {csv_filepath}")

# Main execution starts here
if __name__ == "__main__":
    # Define paths and parameters
    output_folder = r'C:\Users\pis05408.PINNACLE\Desktop\Suraj\ObjectDetectionFlorPlan\test'
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')

    # Load your target image and templates
    image_path = r'data\input_file\LEVEL1_POWER_PLAN_page_0001.jpg'
    template_folder =  r'data\rotation'

    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    if not os.path.exists(template_folder):
        raise FileNotFoundError(f"Template folder not found at path: {template_folder}")

    # Define scales, threshold, and icon name
    scales = np.arange(0.6, 1.2, 0.1)
    threshold = 0.8  # Adjust based on experimentation
    icon_name = 'wall_mounted_duplex_receptacle'

    print('Running OpenCV detection...')

    # Capture detected coordinates (bounding boxes)
    detected_coordinates = capture_detected_coordinates(image_path, template_folder, scales, threshold)

    # Mark the detected objects on the image
    marked_image, object_count = mark_the_objects(image_path, detected_coordinates)

    # Save the marked image 
    save_the_image(marked_image, object_count, output_folder, icon_name)
