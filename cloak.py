"""
Invisibility Cloak using Python + OpenCV
Author: Munazza
Description:
    This project creates the Harry Potter invisibility cloak effect.
    It detects a blue cloak and replaces that area with the background.
"""

import cv2
import numpy as np
import time

def capture_background(cap, frames=30):
    """Capture background image without the subject."""
    background = 0
    for i in range(frames):
        ret, background = cap.read()
        if not ret:
            continue
    return np.flip(background, axis=1)

def invisibility_cloak():
    # Start webcam
    cap = cv2.VideoCapture(0)
    time.sleep(2)   # Allow camera to warm up

    # Capture background
    print("[INFO] Capturing background... Please step out of frame.")
    background = capture_background(cap)
    print("[INFO] Background captured successfully!")

    print("[INFO] Starting invisibility effect. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally (mirror effect)
        frame = np.flip(frame, axis=1)

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for BLUE cloak
        lower_blue = np.array([94, 80, 2])
        upper_blue = np.array([126, 255, 255])

        # Create mask for blue
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Refine mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Inverse mask for non-cloak area
        mask_inv = cv2.bitwise_not(mask)

        # Apply masks
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
        final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

        # Show result
        cv2.imshow("Invisibility Cloak (Blue)", final_output)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    invisibility_cloak()

