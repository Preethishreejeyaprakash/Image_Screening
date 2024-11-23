import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use the proper format for the video path on Windows
video_path = r"C:\Users\USER\Downloads\flask.mp4"  # Video path

cap = cv2.VideoCapture(video_path)

# Define video writer to save the output
output_path = 'output_video.avi'  # Output video file path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Person and vehicle class labels (YOLOv5)
person_class = 0  # "person" class in YOLO
vehicle_classes = [2, 3, 5, 7, 6]  # "car", "bus", "truck", "motorcycle" classes in YOLO

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame if needed (optional)
    frame = cv2.resize(frame, (1000, 650))

    # Perform detection
    results = model(frame)
    df = results.pandas().xyxy[0]

    # Split the frame into left and right sections
    mid_x = frame.shape[1] // 2
    left_frame = frame[:, :mid_x]
    right_frame = frame[:, mid_x:]

    # Draw bounding boxes on the left (green for persons and vehicles)
    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]
        class_id = int(df['class'][ind])
        confidence = df['confidence'][ind]

        # Check if the detected object is a person or a vehicle
        if class_id == person_class or class_id in vehicle_classes:
            # Display confidence scores
            label_with_confidence = f"{label} {confidence*100:.1f}%"

            # Determine if the object is on the left or right side of the frame
            if x2 < mid_x:  # Left side
                # Draw a green rectangle for persons and vehicles
                cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Make the text smaller and bolder
                cv2.putText(left_frame, label_with_confidence, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)
            else:  # Right side
                # Draw a red rectangle for other objects on the right side
                cv2.rectangle(right_frame, (x1 - mid_x, y1), (x2 - mid_x, y2), (0, 0, 255), 2)
                # Make the text smaller and bolder
                cv2.putText(right_frame, label_with_confidence, (x1 - mid_x, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 3)

    # Combine left and right frames back together
    frame = cv2.hconcat([left_frame, right_frame])

    # Display the frame
    cv2.imshow('Video', frame)

    # Write the frame to the output video file
    out.write(frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
