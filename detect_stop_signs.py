# import cv2

# # Load the stop sign cascade
# cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')

# # Open the video file
# video = cv2.VideoCapture('How to stop at a STOP SIGN.mp4')

# # Check if video opened successfully
# if not video.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Optional: Create a window to show the output
# cv2.namedWindow('Stop Sign Detection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Stop Sign Detection', 640, 480)

# while True:
#     # Read one frame from the video
#     ret, frame = video.read()

#     if not ret:
#         break  # End of video

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect stop signs
#     stop_signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     # Draw rectangles around detected stop signs
#     for (x, y, w, h) in stop_signs:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

#     # Show the frame
#     cv2.imshow('Stop Sign Detection', frame)

#     # Exit if ESC key is pressed
#     if cv2.waitKey(10) & 0xFF == 27:
#         break

# # Cleanup
# video.release()
# cv2.destroyAllWindows()


# import cv2

# # --------------------------
# # Parameters (adjust these)
# # --------------------------
# KNOWN_WIDTH = 0.75  # meters (standard stop sign width ~ 75 cm)
# FOCAL_LENGTH = 800  # to be calibrated for your camera

# # Load Haar Cascade for stop sign
# cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')

# # Open the video
# video = cv2.VideoCapture('how-to-stop-at-a-stop-sign_vICt3C3s.mp4')

# if not video.isOpened():
#     print("Error: Could not open video.")
#     exit()

# cv2.namedWindow('Stop Sign Detection', cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow('Stop Sign Detection', 640, 480)

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break  # End of video

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect stop signs
#     stop_signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     closest_distance = None

#     for (x, y, w, h) in stop_signs:
#         # Estimate distance (formula: (Real Width * Focal Length) / Perceived Width)
#         distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

#         # Track the closest stop sign
#         if closest_distance is None or distance < closest_distance:
#             closest_distance = distance
#             closest_box = (x, y, w, h)

#     # Draw only the closest stop sign
#     if closest_distance is not None:
#         (x, y, w, h) = closest_box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
#         cv2.putText(frame, f"STOP SIGN: {closest_distance:.2f} m", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     # Show the frame
#     cv2.imshow('Stop Sign Detection', frame)

#     # Exit on ESC
#     if cv2.waitKey(10) & 0xFF == 27:
#         break

# # Cleanup
# video.release()
# cv2.destroyAllWindows()








import cv2

# --------------------------
# Parameters (adjust these)
# --------------------------
KNOWN_WIDTH = 0.75  # meters (standard stop sign width ~ 75 cm)
FOCAL_LENGTH = 800  # to be calibrated for your camera

# Desired display size (for scaling video window)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 400

# Load Haar Cascade for stop sign
cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')

# Open the video
video = cv2.VideoCapture('How to stop at a STOP SIGN.mp4')

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('Stop Sign Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        break  # End of video

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect stop signs
    stop_signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    closest_distance = None
    closest_box = None

    for (x, y, w, h) in stop_signs:
        # Estimate distance (formula: (Real Width * Focal Length) / Perceived Width)
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

        # Track the closest stop sign
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_box = (x, y, w, h)

    # Draw only the closest stop sign
    if closest_distance is not None:
        (x, y, w, h) = closest_box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"STOP SIGN: {closest_distance:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

    # --------------------------
    # Resize video to fit window
    # --------------------------
    h, w = frame.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Show the resized frame
    cv2.imshow('Stop Sign Detection', resized_frame)

    # Exit on ESC
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Cleanup
video.release()
cv2.destroyAllWindows()

























