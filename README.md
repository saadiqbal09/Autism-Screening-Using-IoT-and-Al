# Autism-Screening-Using-IoT-and-Al

##  Final Year Project | B.E. Industrial Internet of Things (IIoT)

**Drishti AI** is a multimodal, AI-driven screening prototype designed to assist in the early identification of Autism Spectrum Disorder (ASD) using **eye-gaze behavior** and **speech pattern analysis**.  
The system integrates **IoT devices**, **Computer Vision**, and **Machine Learning** to provide an objective, low-cost, and non-invasive screening approach.

> **Disclaimer:** This project is a *screening and research prototype*, not a medical diagnostic tool. Final diagnosis must be performed by qualified healthcare professionals.

---

##  Problem Statement
Traditional ASD screening methods are:
- Subjective
- Time-consuming
- Dependent on expert availability

This leads to **delayed diagnosis**, especially in rural or resource-limited settings.

**Drishti AI addresses this gap** by converting behavioral cues (eye contact & speech fluency) into quantifiable AI-based metrics using commonly available hardware like webcams and microphones.

---

##  System Overview

###  Modalities Used
1. **Visual Analysis (Eye-Tracking)**
   - MediaPipe Face Mesh
   - Gaze fixation & attention analysis

2. **Speech Analysis**
   - OpenAI Whisper (Transformer-based ASR)
   - Detection of:
     - Repetitions
     - Blocks (pauses)
     - Prolongations
     - Response delay

3. **Feature Fusion & Prediction**
   - Rule-based logic (prototype)
   - Mimics Random Forest decision-making
   - Outputs ASD Risk Level:
     - Low
     - Moderate
     - High

---

## Technologies Used

- Python
- Tkinter (GUI)
- OpenAI Whisper (Speech Recognition)
- MediaPipe (planned – eye tracking)
- NumPy, SciPy
- SoundDevice
- Machine Learning (Random Forest – conceptual)

## Project Structure
Autism-Screening-Using-IoT-and-AI
│
├── code/
│ ├── drishti_speech_app.py
│ ├── requirements.txt
│
├── docs/
│ ├── Project_Report.pdf
│ ├── Project_Presentation.pptx
│
├── README.md
└── LICENSE
