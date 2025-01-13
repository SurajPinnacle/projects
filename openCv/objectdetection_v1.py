import cv2
import numpy as np
import os
import csv

def preprocess_image(image_path):
    """
    Preprocess the image by converting it to grayscale.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check the file path or file integrity.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def check_overlap(detected_points, new_box, overlap_threshold=0.3):
    """
    Check if a new bounding box overlaps with existing detections.

    Args:
        detected_points (list): List of existing bounding boxes.
        new_box (list): Coordinates of the new bounding box [x1, y1, x2, y2].
        overlap_threshold (float): IoU threshold to determine overlap.

    Returns:
        bool: True if overlap exists, False otherwise.
    """
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


def match_template(img_gray, template, threshold):
    """
    Perform template matching on the image.

    Args:
        img_gray (np.ndarray): Grayscale image.
        template (np.ndarray): Template image.
        threshold (float): Threshold to consider a match.

    Returns:
        tuple: Locations where the template matches the image.
    """
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    return np.where(result >= threshold)


def load_templates_from_folder(folder_path):
    """
    Load all templates from a given folder.

    Args:
        folder_path (str): Path to the folder containing templates.

    Returns:
        list: List of template file paths.
    """
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]


def capture_detected_coordinates(image_path, template_folder, scales, threshold):
    """
    Capture the coordinates of detected objects in the image using templates.

    Args:
        image_path (str): Path to the image file.
        template_folder (str): Folder containing template images.
        scales (list): List of scales to search for.
        threshold (float): Template matching threshold.

    Returns:
        list: List of detected bounding boxes [x1, y1, x2, y2].
    """
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
            scaled_w, scaled_h = int(original_w * scale), int(original_h * scale)
            if scaled_w <= 0 or scaled_h <= 0:
                continue  # Skip invalid sizes
            template_resized = cv2.resize(template, (scaled_w, scaled_h))

            loc = match_template(img_gray, template_resized, threshold)
            w_rotated, h_rotated = template_resized.shape[::-1]

            for pt in zip(*loc[::-1]):
                x1, y1 = pt
                x2, y2 = x1 + w_rotated, y1 + h_rotated
                new_box = [x1, y1, x2, y2]

                if not check_overlap(detected_coordinates, new_box):
                    detected_coordinates.append(new_box)

    return detected_coordinates


def mark_the_objects(image_path, detected_coordinates):
    """
    Mark the detected objects on the image.

    Args:
        image_path (str): Path to the image file.
        detected_coordinates (list): List of bounding boxes.

    Returns:
        tuple: The image with marked detections and the object count.
    """
    img = cv2.imread(image_path)
    for coord in detected_coordinates:
        x1, y1, x2, y2 = coord
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img, len(detected_coordinates)


def save_the_image(img, object_count, output_folder, icon_name, update_csv=True):
    """
    Save the marked image and update the CSV file with the detection count.

    Args:
        img (np.ndarray): The image with marked detections.
        object_count (int): The number of detected objects.
        output_folder (str): The folder to save the output.
        icon_name (str): The name to associate with the detections.
        update_csv (bool): Flag to determine whether to update the CSV file.

    Returns:
        None
    """
    filename = f"{icon_name}_{object_count}_detections.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, img)

    csv_filepath = os.path.join(output_folder, 'detected_objects.csv')
    file_exists = os.path.isfile(csv_filepath)

    if update_csv:
        with open(csv_filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                csv_writer.writerow(['icon_name', 'object_count'])

            csv_writer.writerow([icon_name, object_count])
            print(f"CSV file updated: {csv_filepath}")

    print(f"Output image saved to: {filepath}")
    print(f"Total Detected icons are: {object_count}")


def detect_icons_in_subfolders(image_path, template_root_folder, output_folder, scales, threshold):
    """
    Detect icons from templates in subfolders and save the result.

    Args:
        image_path (str): Path to the image file.
        template_root_folder (str): Root folder containing template subfolders.
        output_folder (str): Folder to save output.
        scales (list): List of scales to search for.
        threshold (float): Template matching threshold.

    Returns:
        list: List of subfolders where icons were detected.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    if not os.path.exists(template_root_folder):
        raise FileNotFoundError(f"Template folder not found at path: {template_root_folder}")

    detected_folders = []
    all_detected_coordinates = []

    for sub_folder in os.listdir(template_root_folder):
        sub_folder_path = os.path.join(template_root_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            print(f"Checking folder: {sub_folder_path}")

            detected_coordinates = capture_detected_coordinates(image_path, sub_folder_path, scales, threshold)

            if detected_coordinates:
                print(f"Detected icons in folder: {sub_folder_path}")
                all_detected_coordinates.extend(detected_coordinates)
                detected_folders.append(sub_folder_path)

    if all_detected_coordinates:
        marked_image, object_count = mark_the_objects(image_path, all_detected_coordinates)
        icon_name = f"detected_icons_{threshold}"
        save_the_image(marked_image, object_count, output_folder, icon_name, update_csv=False)

    return detected_folders

# Main execution starts here
if __name__ == "__main__":
    # Define paths and parameters
    output_folder = r'C:\Users\pis05408.PINNACLE\Desktop\Suraj\ObjectDetectionFlorPlan\test'
    os.makedirs(output_folder, exist_ok=True)
    print(f'Output folder: {output_folder}')

    # Load your target image and templates
    image_path = r'data\legend.jpg'
    template_folder =  r'data\template'

    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    if not os.path.exists(template_folder):
        raise FileNotFoundError(f"Template folder not found at path: {template_folder}")

    # Define scales, threshold, and icon name
    scales = np.arange(0.6, 1.2, 0.1)
    threshold = 0.75  # Adjust based on experimentation

    icon_name = f'legend_{threshold}'

    print('Running OpenCV detection...')

    detected_icons = detect_icons_in_subfolders(image_path, template_folder, output_folder, scales, threshold)
    print(detected_icons)
    print(len(detected_icons))

    # # Capture detected coordinates (bounding boxes)
    # detected_coordinates = capture_detected_coordinates(image_path, template_folder, scales, threshold)

    # # Mark the detected objects on the image
    # marked_image, object_count = mark_the_objects(image_path, detected_coordinates)

    # # Save the marked image 
    # save_the_image(marked_image, object_count, output_folder, icon_name)
