import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Use the proper format for the video path on Windows
video_path = "video.mp4"  # Use a raw string with 'r' to avoid escape issues

cap = cv2.VideoCapture(video_path)

# Define video writer to save the output
output_path = 'output_video.avi'  # Output video file path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

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

    # Draw bounding boxes and labels on the frame
    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

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