# 🔍 YOLOv5 Object Detection Web App

A powerful web application for real-time object detection using YOLOv5 deep learning model, built with Streamlit.

## ✨ Features

- **Image Detection**: Upload images and detect objects in real-time
- **Video Detection**: Process videos and detect objects frame-by-frame
- **Multiple Model Sizes**: Choose from YOLOv5s, YOLOv5m, YOLOv5l, or YOLOv5x
- **Adjustable Confidence**: Set custom confidence thresholds for detections
- **Visual Results**: See detection results with bounding boxes and labels
- **Detection Statistics**: View detailed object counts and confidence scores
- **Interactive UI**: User-friendly Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/akhil542354/cv-object-detection-webapp.git
cd cv-object-detection-webapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository
6. Set the main file path to `app.py`
7. Click "Deploy"

## 📦 Dependencies

- **streamlit**: Web application framework
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision library for PyTorch
- **opencv-python-headless**: Computer vision library (headless for cloud deployment)
- **Pillow**: Image processing library
- **numpy**: Numerical computing library
- **pandas**: Data manipulation library

## 🎯 How to Use

1. **Select Input Type**: Choose between Image or Video detection
2. **Adjust Settings**: 
   - Select model size (smaller = faster, larger = more accurate)
   - Set confidence threshold (0.0 - 1.0)
3. **Upload File**: Upload your image (JPG, PNG) or video (MP4, AVI, MOV)
4. **View Results**: See detected objects with bounding boxes and labels
5. **Analyze**: Review detection statistics and object counts

## 🧠 Model Information

### YOLOv5 Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| YOLOv5s | Smallest | Fastest | Good |
| YOLOv5m | Medium | Fast | Better |
| YOLOv5l | Large | Slower | Great |
| YOLOv5x | Largest | Slowest | Best |

### Detectable Objects

YOLOv5 can detect **80 different object classes** from the COCO dataset, including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, train, airplane, boat
- **Animals**: dog, cat, bird, horse, cow, elephant, bear, zebra, giraffe
- **Sports**: tennis racket, baseball bat, skateboard, surfboard, sports ball
- **Indoor Objects**: chair, couch, bed, dining table, tv, laptop, mouse, keyboard
- **Kitchen Items**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, pizza, donut, cake
- And many more!

## 🛠️ Technical Details

- **Framework**: Streamlit
- **Model**: YOLOv5 from Ultralytics
- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV
- **Image Processing**: Pillow, NumPy

## 📝 Project Structure

```
cv-object-detection-webapp/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🔧 Configuration

You can modify the following settings in the app:

- **Confidence Threshold**: Adjust the minimum confidence score for detections (0.0 - 1.0)
- **Model Size**: Select from YOLOv5s, YOLOv5m, YOLOv5l, or YOLOv5x

## 🌐 Deployment

This app is ready for deployment on:
- **Streamlit Cloud** (recommended)
- **Heroku**
- **Google Cloud Platform**
- **AWS**
- **Azure**

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or feedback, please open an issue in this repository.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection model
- [Streamlit](https://streamlit.io/) for the web framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

---

**Built with ❤️ using Streamlit and YOLOv5**
