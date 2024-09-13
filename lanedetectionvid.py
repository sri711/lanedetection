import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Define a region of interest (ROI) polygon
    mask = np.zeros_like(edges)
    height, width = image.shape[:2]
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    # Draw detected lines on a blank image
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    # Combine the line image with the original image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result

# Read the input video
cap = cv2.VideoCapture(r'C:\Users\sriprakash\OneDrive\Desktop\aiml\reference\code\computer vision\test_video.mp4')

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()     #ret- read or not
    if not ret:
        break
    
    # Detect lanes in the current frame
    processed_frame = detect_lanes(frame)
    
    # Write the processed frame to the output video
    out.write(processed_frame)
    
    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
