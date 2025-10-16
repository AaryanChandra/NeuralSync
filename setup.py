#!/usr/bin/env python3
"""
NeuralSync Setup Script
Automated installation and setup for the AI Emotion Music Engine
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "face/requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_webcam():
    """Check if webcam is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Webcam detected and working")
                return True
        print("⚠️  Webcam not detected or not working")
        return False
    except ImportError:
        print("⚠️  OpenCV not installed yet, skipping webcam check")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ["face/static", "face/templates", "face/data/train", "face/data/test"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directory structure created")

def check_model_file():
    """Check if model file exists"""
    model_path = "face/model.h5"
    if os.path.exists(model_path):
        print("✅ Pre-trained model found")
        return True
    else:
        print("⚠️  Model file not found. You may need to train the model first.")
        return False

def main():
    """Main setup function"""
    print("🎵 NeuralSync: AI Emotion Music Engine Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check webcam
    check_webcam()
    
    # Check model
    check_model_file()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: cd face && python app.py")
    print("2. Open: http://127.0.0.1:5000")
    print("3. Allow camera permissions")
    print("4. Start making expressions!")
    
    print("\n🔧 Troubleshooting:")
    print("- If webcam issues: Check camera permissions")
    print("- If model errors: Run train.py to train your own model")
    print("- If port issues: Change port in app.py")

if __name__ == "__main__":
    main()
