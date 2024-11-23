Here’s a **README** file for your project:

---

# Real-Time Object Detection with YOLOv5  

## Description  
This project demonstrates real-time object detection using the YOLOv5 model and OpenCV. It processes a video file, detects objects in each frame, and saves the annotated video with bounding boxes and labels.

---

## Features  
- **Object Detection**: Detects multiple objects in a video frame using YOLOv5.  
- **Bounding Boxes**: Draws bounding boxes and labels for detected objects.  
- **Video Output**: Saves the processed video with annotations.  
- **Real-Time Display**: Displays video frames with detected objects during processing.  

---

## Prerequisites  

### Requirements:  
- Python 3.7 or later  
- OpenCV  
- PyTorch  
- YOLOv5 Model  

### Installation:  
1. Install Python dependencies:  
   ```bash
   pip install torch torchvision opencv-python pandas
   ```
2. Download or clone the YOLOv5 repository using PyTorch Hub automatically in the code.  

---

## Usage  

1. **Prepare a Video File**:  
   - Place the video file in the project directory.  
   - Replace `video_path` with your video file's name or path (e.g., `"video.mp4"`).  

2. **Run the Script**:  
   - Execute the script using:  
     ```bash
     python object_detection.py
     ```  

3. **View Results**:  
   - Annotated frames will be displayed in a window.  
   - Press **'q'** to stop the video display.  
   - Annotated video will be saved as `output_video.avi`.  

---

## Code Explanation  

1. **Model Loading**:  
   The YOLOv5 model is loaded using PyTorch Hub.  
   ```python
   model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
   ```  

2. **Video Processing**:  
   - Video frames are read, resized, and passed to the YOLOv5 model for detection.  
   - Detected objects are highlighted with bounding boxes and labels using OpenCV.  

3. **Output Generation**:  
   Annotated frames are written to a video file using `cv2.VideoWriter`.  

---

## Output Example  

- **Input**:  
  Original video (`video.mp4`).  
- **Output**:  
  Annotated video with bounding boxes and labels saved as `output_video.avi`.  

---

## Notes  

- Ensure the video file path is correct for your system. Use a raw string (`r"path"`) to avoid escape sequence issues on Windows.  
- Adjust the frame size or detection model as needed for better performance.  
- YOLOv5's detection capabilities depend on the model version and pretraining dataset.  

---

## Acknowledgements  

- **YOLOv5**: [Ultralytics YOLOv5 Repository](https://github.com/ultralytics/yolov5)  
- **OpenCV**: [OpenCV Documentation](https://opencv.org/)  
- **PyTorch Hub**: [PyTorch Hub](https://pytorch.org/hub/)  

---

## License  

This project is open-source and can be modified for personal or academic purposes.  

## Screenshot and Output



![Screenshot 2024-11-23 154651](https://github.com/user-attachments/assets/e947559f-7239-4772-907d-2361ccab853a)

