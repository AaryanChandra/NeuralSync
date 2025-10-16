# 🎵 NeuralSync: AI-Powered Emotion Music Engine

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.0-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.1.2-green.svg)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-red.svg)](https://opencv.org)

A real-time facial emotion recognition system that recommends personalized music based on detected emotions using deep learning and computer vision.

## 🎯 Features

- **Real-time Emotion Detection**: 7 emotion classes (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised)
- **Live Music Recommendations**: Personalized song suggestions based on detected emotions
- **Interactive Web Interface**: Cyberpunk-themed UI with live video feed
- **Audio Previews**: 30-second music previews via iTunes API
- **Smart Confirmation System**: Prevents rapid emotion switching with user confirmation
- **Performance Optimized**: Multi-threaded processing with prediction smoothing

## 🏗️ System Architecture

```
Webcam Feed → Face Detection → Emotion Classification → Music Recommendation → Web Display
```

### Core Components:
- **Frontend**: HTML5, CSS3, JavaScript with cyberpunk styling
- **Backend**: Flask web server with RESTful APIs
- **Computer Vision**: OpenCV + TensorFlow CNN model
- **Music System**: Pandas + CSV datasets + iTunes integration

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Webcam
- Internet connection (for music previews)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AaryanChandra/NeuralSync.git
   cd NeuralSync
   ```

2. **Install dependencies**
   ```bash
   pip install -r face/requirements.txt
   ```

3. **Run the application**
   ```bash
   cd face
   python app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://127.0.0.1:5000`
   - Allow camera permissions
   - Start making expressions!

## 📁 Project Structure

```
NeuralSync/
├── face/                          # Main application directory
│   ├── app.py                     # Flask web server
│   ├── camera.py                  # Computer vision engine
│   ├── train.py                   # CNN model training script
│   ├── Spotipy.py                 # Spotify integration
│   ├── utils.py                   # Utility functions
│   ├── model.h5                   # Trained CNN weights
│   ├── haarcascade_frontalface_default.xml  # Face detection
│   ├── requirements.txt           # Python dependencies
│   ├── templates/
│   │   └── index.html             # Main web interface
│   ├── static/
│   │   └── style.css              # UI styling
│   └── *.csv                      # Music datasets per emotion
├── README.md                      # This file
└── static/                        # Additional static files
```

## 🎵 Music Recommendation System

The system maps detected emotions to curated music datasets:

| Emotion | Dataset | Description |
|---------|---------|-------------|
| 😠 Angry | `angry.csv` | High-energy, aggressive tracks |
| 🤢 Disgusted | `disgusted.csv` | Dark, intense music |
| 😨 Fearful | `fearful.csv` | Atmospheric, suspenseful tracks |
| 😊 Happy | `happy.csv` | Upbeat, cheerful songs |
| 😐 Neutral | `neutral.csv` | Balanced, ambient music |
| 😢 Sad | `sad.csv` | Melancholic, emotional tracks |
| 😲 Surprised | `surprised.csv` | Dynamic, unexpected music |

## 🧠 Technical Details

### CNN Model Architecture
- **Input**: 48x48 grayscale images
- **Architecture**: 7-layer CNN with Conv2D, MaxPooling, and Dense layers
- **Classes**: 7 emotions
- **Optimization**: Adam optimizer with categorical crossentropy loss

### Performance Optimizations
- **Video Resolution**: 640x480 for optimal performance
- **Prediction Throttling**: Every 3rd frame to reduce CPU load
- **Prediction Smoothing**: 9-frame sliding window for stability
- **Cached Recommendations**: Only update when emotion changes

### API Endpoints
- `GET /` - Main dashboard
- `GET /video_feed` - Live video stream
- `GET /t` - Music recommendations JSON
- `GET /emotion` - Current emotion status

## 🎨 User Interface

The web interface features:
- **Live Video Feed**: Real-time emotion detection display
- **Status Indicator**: Shows current detected emotion or "No face detected"
- **Music Player**: Top 3 track previews with iTunes integration
- **Confirmation Modals**: Prevents unwanted emotion switching
- **Cyberpunk Theme**: Neon colors, animations, and futuristic styling

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Disable TensorFlow optimizations for debugging
export TF_ENABLE_ONEDNN_OPTS=0
```

### Customization
- **Music Datasets**: Replace CSV files with your own music collections
- **Model**: Retrain with `train.py` using your own emotion datasets
- **UI Theme**: Modify `static/style.css` for different styling

## 📊 Performance Metrics

- **Face Detection**: ~30 FPS on modern hardware
- **Emotion Prediction**: ~10 FPS (throttled for stability)
- **UI Updates**: 500ms polling interval
- **Memory Usage**: ~200MB RAM (model + video processing)

## 🛠️ Development

### Training Your Own Model
1. Prepare emotion datasets in `data/train/` and `data/test/`
2. Run `python train.py` to train the CNN model
3. Replace `model.h5` with your trained weights

### Adding New Emotions
1. Update `emotion_dict` in `camera.py`
2. Add corresponding CSV files
3. Retrain the model with additional classes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**AaryanChandra**
- GitHub: [@AaryanChandra](https://github.com/AaryanChandra)
- Project: NeuralSync AI Emotion Music Engine

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow team for deep learning framework
- Flask team for web framework
- iTunes API for music previews
- Emotion recognition research community

## 📞 Support

If you encounter any issues:
1. Check the [Issues](https://github.com/AaryanChandra/NeuralSync/issues) page
2. Create a new issue with detailed description
3. Include system specifications and error logs

---

⭐ **Star this repository if you found it helpful!**