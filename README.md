# Sign Language Gesture Recognition System🫶🏻

Welcome to the Sign Language Gesture Recognition System! This project leverages computer vision and deep learning to bridge communication gaps by recognizing sign language gestures in real time. Whether you are learning sign language or enhancing accessibility, this system is designed for you. 🤝

## Features

- **Real-Time Recognition:** Capture and interpret gestures live through your webcam. 📹
- **Multi-Gesture Support:** Recognize various gestures, including "hello," "thanks," and "I love you." ✋
- **Visual Feedback:** Instant visual representation of recognized gestures on the screen. 👀
- **Audio Feedback:** Text-to-speech functionality that audibly conveys recognized gestures. 🔊
Technologies Used

**This project utilizes several powerful technologies:**

- **Python:** The primary programming language. 🐍
- **OpenCV:** Advanced image processing and computer vision capabilities. 🖼️
- **MediaPipe:** Efficient detection and tracking of hand landmarks. ✋
- **TensorFlow:** Framework for building and training the deep learning model. 📊
- **pyttsx3:** Library for voice output of recognized gestures. 🎤
Project Structure
-----

### Here’s an overview of the project layout:

```
Copy code
/SignLanguageGestureRecognition
├── MP_Data/                # Collected gesture data
│   ├── hello/
│   ├── thanks/
│   └── i_love_you/
├── training.py             # Data collection and preprocessing
├── execution.py            # Gesture recognition execution
└── README.md               # Project documentation
```
## Installation

**Ready to get started? Follow these simple steps:**

1. **Clone the Repository:**
```
git clone https://github.com/yourusername/SignLanguageGestureRecognition.git
cd SignLanguageGestureRecognition
```
2. **Install Required Packages:** Ensure you have Python installed, then run:
```
pip install opencv-python mediapipe numpy tensorflow pyttsx3
```
## Usage

- **Data Collection:** Start by capturing gesture data. Run the following command:
```
python training.py
```
Follow the on-screen instructions to build your dataset.
- **Gesture Recognition:** Once your data is ready, recognize gestures in real time:
```
python execution.py
```



## Acknowledgments

Thanks to the technologies that made this project possible:

- **MediaPipe** for its landmark detection capabilities. 👏
- **OpenCV** for its robust image processing functionalities. 🔍
- **TensorFlow**for providing a powerful deep learning framework. ⚙️
- **pyttsx3** for enabling text-to-speech functionality. 🎶
