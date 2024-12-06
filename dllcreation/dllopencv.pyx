import cv2
cimport cython
import numpy as np
import os


@cython.boundscheck(False)
@cython.wraparound(False)
# Function to preprocess the image (gray scaling)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check the file path or file integrity.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

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

# Function to detect icons in an image
def detect_icons(image_path, template_paths, scales, threshold, output_folder, icon_name):
    # Load and preprocess the main image
    img, img_gray = preprocess_image(image_path)

    # Create a copy of the image for drawing detections
    img_detected = img.copy()
    detected_points = []  # To store detected bounding boxes
    count_detected = 0

    # Iterate over each template
    for template_path in template_paths:
        # Load and preprocess the template
        template = cv2.imread(template_path, 0)  # Load as grayscale
        original_w, original_h = template.shape[::-1]

        # Iterate over scales
        for scale in scales:
            # Resize the template according to the scale
            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)
            if scaled_w <= 0 or scaled_h <= 0:
                continue  # Skip invalid sizes
            template_resized = cv2.resize(template, (scaled_w, scaled_h))

            # Prepare a list to hold all rotated templates
            templates = []

            # Resize and rotate the template
            for i in range(4):
                if i == 0:
                    # Original template
                    template_rotated = template_resized.copy()
                else:
                    # Rotate 90 degrees clockwise
                    template_rotated = cv2.rotate(template_rotated, cv2.ROTATE_90_CLOCKWISE)
                templates.append(template_rotated)

            # Perform template matching for each rotated template
            for template_rotated in templates:
                w_rotated, h_rotated = template_rotated.shape[::-1]
                loc = match_template(img_gray, template_rotated, threshold)

                # Iterate over all matched locations
                for pt in zip(*loc[::-1]):  # Switch columns and rows
                    x1, y1 = pt
                    x2, y2 = x1 + w_rotated, y1 + h_rotated
                    new_box = [x1, y1, x2, y2]

                    # Check for overlap with existing detections
                    if not check_overlap(detected_points, new_box):
                        # Draw rectangle on the detected image
                        cv2.rectangle(img_detected, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        detected_points.append(new_box)
                        count_detected += 1

    print(f"Total detections: {count_detected}")

    # Save the detected image
    filename = f"{icon_name}_CNT_{count_detected}.jpg"
    filepath = os.path.join(output_folder, filename)
    print(f"Output image saved to: {filepath}")
    cv2.imwrite(filepath, img_detected)

# Main execution starts here
if __name__ == "__main__":
    # Define paths and parameters
    output_folder = 'C:/Users/pis05408.PINNACLE/Desktop/Suraj/ObjectDetectionFlorPlan/test'
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')

    # Load your target image and templates
    image_path = '../openCv/data/input_file/LEVEL1_POWER_PLAN_page_0001.jpg'
    template_paths = ['../openCv/data/d_icon.jpg']

    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    # Define scales, threshold, and icon name
    scales = np.arange(0.6, 1.2, 0.1)
    threshold = 0.7  # Adjust based on experimentation
    icon_name = 'ddl_test'

    print('running the OpenCv')

    # Call the detection function
    detect_icons(image_path, template_paths, scales, threshold, output_folder, icon_name)
