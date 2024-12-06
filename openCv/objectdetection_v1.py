import cv2
import numpy as np
import os
import csv

# Constants for file types and threshold
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg')
DEFAULT_OVERLAP_THRESHOLD = 0.3
DEFAULT_THRESHOLD = 0.8
DEFAULT_SCALES = np.arange(0.6, 1.2, 0.1)
CSV_HEADER = ['icon_name', 'object_count']

# Function to preprocess the image (gray scaling)
def preprocess_image(image_path):
    """Preprocess the input image and convert it to grayscale."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check the file path or file integrity.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Function to check for overlap between detected icons
def check_overlap(detected_points, new_box, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD):
    """Check if the new bounding box overlaps with any of the detected boxes."""
    x1, y1, x2, y2 = new_box
    for existing_box in detected_points:
        xa1, ya1, xa2, ya2 = existing_box
        inter_x1 = max(x1, xa1)
        inter_y1 = max(y1, ya1)
        inter_x2 = min(x2, xa2)
        inter_y2 = min(y2, ya2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box_area = (x2 - x1) * (y2 - y1)
        existing_area = (xa2 - xa1) * (ya2 - ya1)
        union_area = box_area + existing_area - inter_area
        if union_area == 0:
            continue
        iou = inter_area / union_area
        if iou > overlap_threshold:
            return True
    return False

# Function to perform template matching on an image
def match_template(img_gray, template, threshold):
    """Perform template matching on the grayscale image."""
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    return np.where(result >= threshold)

# Function to load all templates from a folder
def load_templates_from_folder(folder_path):
    """Load all image templates from a folder."""
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
            if filename.endswith(VALID_IMAGE_EXTENSIONS)]

# Function to capture detected object coordinates (bounding boxes)
def capture_detected_coordinates(image_path, template_folder, scales, threshold):
    """Capture coordinates of detected objects using template matching."""
    img_gray = preprocess_image(image_path)
    detected_coordinates = []
    template_paths = load_templates_from_folder(template_folder)

    for template_path in template_paths:
        template = cv2.imread(template_path, 0)
        if template is None:
            print(f"Skipping invalid template: {template_path}")
            continue
        original_w, original_h = template.shape[::-1]

        for scale in scales:
            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)
            if scaled_w <= 0 or scaled_h <= 0:
                continue
            template_resized = cv2.resize(template, (scaled_w, scaled_h))
            loc = match_template(img_gray, template_resized, threshold)

            for pt in zip(*loc[::-1]):
                x1, y1 = pt
                x2, y2 = x1 + scaled_w, y1 + scaled_h
                new_box = [x1, y1, x2, y2]

                if not check_overlap(detected_coordinates, new_box):
                    detected_coordinates.append(new_box)

    return detected_coordinates

# Function to mark the detected objects on the image and return the object count
def mark_the_objects(image_path, detected_coordinates):
    """Mark the detected bounding boxes on the image."""
    img = cv2.imread(image_path)
    for coord in detected_coordinates:
        x1, y1, x2, y2 = coord
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img, len(detected_coordinates)

# Function to save the image and update the CSV file with the object count
def save_the_image(img, object_count, output_folder, icon_name):
    """Save the detected image and update the CSV file with object count."""
    filename = f"{icon_name}_{object_count}_detections.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, img)

    csv_filepath = os.path.join(output_folder, 'detected_objects.csv')
    file_exists = os.path.isfile(csv_filepath)

    with open(csv_filepath, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(CSV_HEADER)
        csv_writer.writerow([icon_name, object_count])

    print(f"Output image saved to: {filepath}")
    print(f"Total Detected icons: {object_count}")
    print(f"CSV file updated: {csv_filepath}")

# Main execution starts here
def main():
    """Main function to run the object detection pipeline."""
    output_folder = r'C:\Users\pis05408.PINNACLE\Desktop\Suraj\ObjectDetectionFlorPlan\test'
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')

    image_path = r'data\input_file\LEVEL1_POWER_PLAN_page_0001.jpg'
    template_folder = r'data\rotation'

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    if not os.path.exists(template_folder):
        raise FileNotFoundError(f"Template folder not found at path: {template_folder}")

    scales = DEFAULT_SCALES
    threshold = DEFAULT_THRESHOLD
    icon_name = 'wall_mounted_duplex_receptacle'

    print('Running OpenCV detection...')

    detected_coordinates = capture_detected_coordinates(image_path, template_folder, scales, threshold)
    marked_image, object_count = mark_the_objects(image_path, detected_coordinates)
    save_the_image(marked_image, object_count, output_folder, icon_name)

if __name__ == "__main__":
    main()
