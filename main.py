import cv2
import os
import numpy as np

def detect_onion_color(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define thresholds for each color
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_pink = np.array([150, 50, 50])
    upper_pink = np.array([170, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Count the number of non-zero pixels in each mask
    red_pixels = cv2.countNonZero(mask_red)
    pink_pixels = cv2.countNonZero(mask_pink)
    white_pixels = cv2.countNonZero(mask_white)

    # Determine the predominant color based on the number of pixels
    max_pixels = max(red_pixels, pink_pixels, white_pixels)
    if max_pixels == red_pixels:
        return "Red"
    elif max_pixels == pink_pixels:
        return "Pink"
    elif max_pixels == white_pixels:
        return "White"
    else:
        return "Unknown"

def count_onion_colors(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to obtain binary mask
    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counters for red, pink, and white onions
    red_count = 0
    pink_count = 0
    white_count = 0

    # Iterate through contours and classify onions based on color
    for contour in contours:
        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI) from the original image
        roi = image[y:y+h, x:x+w]

        # Classify the color of the onion using the detect_onion_color function
        color = detect_onion_color(roi)

        # Update the count based on the color classification
        if color == "Red":
            red_count += 1
        elif color == "Pink":
            pink_count += 1
        elif color == "White":
            white_count += 1

    return red_count, pink_count, white_count

def main():
    # Directory containing onion images
    project_dir = "../ONION DETECTION/"
    image_file = os.path.join(project_dir, "onions.jpg")

    # Check if the image file exists
    if not os.path.isfile(image_file):
        print("Error: Image file not found:", image_file)
        return

    # Read the image from file
    image = cv2.imread(image_file)

    if image is None:
        print("Error: Failed to read image:", image_file)
        return

    # Display the image
    cv2.imshow('Onion Detection', image)

    # Count the number of onions of each color
    red_count, pink_count, white_count = count_onion_colors(image)
    print("Number of Red Onions:", red_count)
    print("Number of Pink Onions:", pink_count)
    print("Number of White Onions:", white_count)

    # Wait for a key press to exit
    cv2.waitKey(0)

    # Close the OpenCV window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()