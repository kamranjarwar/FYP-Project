import numpy as np
import easyocr
import cv2

def get_car(license_plate, track_ids):
    """
    Match the license plate to the corresponding car using tracking IDs.
    Args:
        license_plate: The bounding box of the detected license plate
        track_ids: List of vehicle tracking IDs
    Returns:
        xcar1, ycar1, xcar2, ycar2, car_id: The bounding box of the matched car and its tracking ID
    """
    x1, y1, x2, y2, score, class_id = license_plate
    # Assuming the license plate detector provides coordinates in (x1, y1, x2, y2)
    for car_id, (xc1, yc1, xc2, yc2, _) in enumerate(track_ids):
        # You can choose the best matching car based on the intersection of bounding boxes, distance, etc.
        if x1 > xc1 and y1 > yc1 and x2 < xc2 and y2 < yc2:  # Simple bounding box check
            return int(xc1), int(yc1), int(xc2), int(yc2), car_id
    return -1, -1, -1, -1, -1  # No match found


def read_license_plate(license_plate_image):
    """
    Use OCR to read text from a license plate image.
    Args:
        license_plate_image: The cropped image of the license plate.
    Returns:
        text: The recognized text from the license plate.
        score: Confidence score of the OCR recognition.
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # 'en' is for English; you can add more languages if needed
    
    # Check the number of channels in the input image
    if len(license_plate_image.shape) == 3:
        # If the image has 3 channels (color), convert it to grayscale
        license_plate_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
    else:
        # If the image is already grayscale (1 channel), use it directly
        license_plate_gray = license_plate_image

    # Optional thresholding to improve OCR
    _, thresh = cv2.threshold(license_plate_gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Perform OCR using EasyOCR
    result = reader.readtext(thresh)
    
    # Extract text and score from OCR results
    text = ' '.join([item[1] for item in result])  # Concatenate all detected text
    score = sum([item[2] for item in result]) / len(result) if result else 0  # Average confidence score
    
    return text.strip(), score
