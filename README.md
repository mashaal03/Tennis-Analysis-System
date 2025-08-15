# 🎾 Tennis Analysis System

## 📌 Introduction
This project analyzes tennis players in a video to measure:
- Player speed
- Ball shot speed
- Number of shots  

It uses **YOLO** for player & ball detection and **CNNs** for tennis court keypoint extraction.  
This hands-on project is perfect for polishing your **machine learning** and **computer vision** skills.  

---

## 📸 Output Example
Here’s a screenshot from one of the output videos:  

![Screenshot](output_videos/Screenshot%202025-08-16%20013050.png)  

---

## 🧠 Models Used
- **YOLOv8** for player detection  
- Fine-tuned **YOLOv8** for tennis ball detection  
- Court keypoint extraction using CNNs  

**Pre-trained Models:**  
- [YOLOv8 Player Detection Model](https://drive.google.com/file/d/1aRoZyZ3thLkhL88AhXYhEgd12ETPKrHW/view?usp=drive_link)  
- [Tennis Court Keypoint Model](https://drive.google.com/file/d/1SYLG4yks_mVoBVEn6E3TltJaIwrFAc7H/view?usp=drive_link)  

---

## 🏋️ Training Notebooks
- [Tennis Ball Detector with YOLO](training/tennis_ball_detector_training.ipynb)  
- [Tennis Court Keypoints with PyTorch](training/tennis_court_keypoints_training.ipynb)  

---

## 📦 Requirements
- Python 3.8  
- `ultralytics`  
- `torch`  
- `pandas`  
- `numpy`  
- `opencv-python`  

---

## 📂 Project Structure
