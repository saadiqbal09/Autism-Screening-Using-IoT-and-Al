import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import sounddevice as sd
from scipy.io.wavfile import write
import torch
from transformers import pipeline
import numpy as np
import threading
import time
from PIL import Image, ImageTk
import string
import os
import warnings

# --- Configuration ---
AUDIO_FILENAME = "child_response.wav"
SAMPLE_RATE = 16000
# Using 'small' model for speed on laptops. Change to 'medium' for better accuracy if you have GPU.
WHISPER_MODEL_ID = "openai/whisper-small" 

# --- Filter Warnings ---
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", message=".*Whisper did not predict an ending timestamp.*")

class DrishtiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drishti AI - Integrated ASD Screening")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")

        # --- State Variables ---
        self.is_recording_audio = False
        self.is_tracking_eyes = False
        self.audio_frames = []
        self.whisper_model = None
        
        # Metrics Storage
        self.gaze_score = 0.0
        self.speech_metrics = {}

        # --- Layout ---
        self.setup_ui()
        
        # --- Load AI Models in Background ---
        self.status_var.set("⏳ Loading AI Models... (This takes a moment)")
        threading.Thread(target=self.load_models, daemon=True).start()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#00d2d3", height=60)
        header.pack(fill="x")
        tk.Label(header, text="DRISHTI AI", font=("Segoe UI", 24, "bold"), bg="#00d2d3", fg="white").pack(pady=5)

        # Main Layout: Split into Left (Video) and Right (Controls/Results)
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # --- Left Side: Video Feed ---
        self.video_frame = tk.Label(main_frame, bg="black", text="Webcam Feed Off", fg="white")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # --- Right Side: Controls ---
        control_panel = tk.Frame(main_frame, bg="#2d3436", width=400)
        control_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Step 1: Eye Tracking
        tk.Label(control_panel, text="Step 1: Visual Attention", font=("Segoe UI", 14, "bold"), bg="#2d3436", fg="#00d2d3").pack(pady=10)
        self.btn_eye_start = tk.Button(control_panel, text="👁️ Start Eye Tracking", command=self.toggle_eye_tracking, bg="#0984e3", fg="white", font=("Segoe UI", 11), width=25)
        self.btn_eye_start.pack(pady=5)
        self.lbl_gaze_result = tk.Label(control_panel, text="Gaze Focus Score: --%", font=("Segoe UI", 12), bg="#2d3436", fg="white")
        self.lbl_gaze_result.pack(pady=5)

        tk.Frame(control_panel, height=2, bg="#636e72").pack(fill="x", pady=15) # Separator

        # Step 2: Audio Analysis
        tk.Label(control_panel, text="Step 2: Speech Response", font=("Segoe UI", 14, "bold"), bg="#2d3436", fg="#00d2d3").pack(pady=10)
        
        # Language Select
        tk.Label(control_panel, text="Select Language:", bg="#2d3436", fg="white").pack()
        self.lang_var = tk.StringVar(value="english")
        self.lang_combo = ttk.Combobox(control_panel, textvariable=self.lang_var, values=["english", "hindi", "marathi", "tamil"], state="readonly")
        self.lang_combo.pack(pady=5)

        self.btn_audio_rec = tk.Button(control_panel, text="🎙️ Start Recording", command=self.toggle_audio_recording, bg="#e17055", fg="white", font=("Segoe UI", 11), width=25, state="disabled")
        self.btn_audio_rec.pack(pady=5)
        
        self.txt_audio_result = tk.Text(control_panel, height=8, width=40, bg="#1e1e1e", fg="#00d2d3", font=("Consolas", 9))
        self.txt_audio_result.pack(pady=5)

        tk.Frame(control_panel, height=2, bg="#636e72").pack(fill="x", pady=15) # Separator

        # Step 3: Final Prediction
        self.lbl_final_risk = tk.Label(control_panel, text="FINAL RISK ASSESSMENT: Pending...", font=("Segoe UI", 14, "bold"), bg="#2d3436", fg="gray")
        self.lbl_final_risk.pack(pady=20)

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(self.root, textvariable=self.status_var, bg="#2d3436", fg="white", anchor="w").pack(side="bottom", fill="x")

        # Configure Grid Weights
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def load_models(self):
        # 1. Load Whisper
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = pipeline(
                "automatic-speech-recognition", 
                model=WHISPER_MODEL_ID, 
                device=device, 
                return_timestamps="word"
            )
            self.status_var.set(f"✅ AI Models Loaded ({device}). System Ready.")
            self.btn_audio_rec.config(state="normal")
        except Exception as e:
            self.status_var.set(f"❌ Error loading model: {e}")

        # 2. Setup MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # --- PART 1: EYE TRACKING LOGIC ---
    def toggle_eye_tracking(self):
        if not self.is_tracking_eyes:
            self.is_tracking_eyes = True
            self.btn_eye_start.config(text="⏹ Stop Tracking", bg="#d63031")
            self.cap = cv2.VideoCapture(0)
            self.gaze_frames_focused = 0
            self.gaze_frames_total = 0
            self.update_video_feed()
        else:
            self.is_tracking_eyes = False
            self.btn_eye_start.config(text="👁️ Start Eye Tracking", bg="#0984e3")
            if self.cap:
                self.cap.release()
            self.video_frame.config(image="")
            
            # Calculate Score
            if self.gaze_frames_total > 0:
                self.gaze_score = (self.gaze_frames_focused / self.gaze_frames_total) * 100
                self.lbl_gaze_result.config(text=f"Gaze Focus Score: {self.gaze_score:.1f}%")
                self.check_final_risk()

    def update_video_feed(self):
        if self.is_tracking_eyes:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                h, w, _ = frame.shape
                is_focused = False

                if results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    
                    # Visual Feedback: Draw Iris
                    (lx, ly), l_rad = cv2.minEnclosingCircle(mesh_points[468:473]) # Left Iris
                    (rx, ry), r_rad = cv2.minEnclosingCircle(mesh_points[473:478]) # Right Iris
                    cv2.circle(frame, (int(lx), int(ly)), int(l_rad), (0, 255, 0), 1)
                    cv2.circle(frame, (int(rx), int(ry)), int(r_rad), (0, 255, 0), 1)

                    # Simple Logic: Is head centered? (Nose tip index 1)
                    nose_x = mesh_points[1][0]
                    if w * 0.35 < nose_x < w * 0.65:
                        is_focused = True
                        cv2.putText(frame, "FOCUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "DISTRACTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update Metrics
                self.gaze_frames_total += 1
                if is_focused:
                    self.gaze_frames_focused += 1

                # Display in TKinter
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            
            self.root.after(10, self.update_video_feed)

    # --- PART 2: AUDIO LOGIC ---
    def toggle_audio_recording(self):
        if not self.is_recording_audio:
            self.is_recording_audio = True
            self.audio_frames = []
            self.btn_audio_rec.config(text="⏹ Stop & Analyze", bg="#d63031")
            threading.Thread(target=self.record_loop, daemon=True).start()
        else:
            self.is_recording_audio = False
            self.btn_audio_rec.config(text="🎙️ Start Recording", bg="#e17055", state="disabled")
            self.status_var.set("Processing Audio... Please Wait.")
            
            # Save and Analyze
            audio_data = np.concatenate(self.audio_frames, axis=0)
            write(AUDIO_FILENAME, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
            threading.Thread(target=self.analyze_audio, daemon=True).start()

    def record_loop(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=self.audio_callback):
            while self.is_recording_audio:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording_audio:
            self.audio_frames.append(indata.copy())

    def analyze_audio(self):
        try:
            lang = self.lang_var.get()
            
            # Use specific generation parameters to stop loops/hallucinations
            generate_kwargs = {
                "language": lang,
                "task": "transcribe",
                "condition_on_prev_tokens": False, # Important to prevent loops
                "temperature": 0.0, # Deterministic output
            }
            
            result = self.whisper_model(
                AUDIO_FILENAME, 
                generate_kwargs=generate_kwargs
            )
            chunks = result["chunks"]
            text = result["text"]
            
            # Calculate Metrics
            response_time = 0.0
            if len(chunks) > 0:
                response_time = chunks[0]['timestamp'][0]
            
            blocks = 0
            prolongations = 0
            repetitions = 0

            for i, chunk in enumerate(chunks):
                word = chunk['text'].strip()
                # Handle edge case where timestamps might be None
                if chunk['timestamp'][0] is None or chunk['timestamp'][1] is None:
                    continue
                    
                duration = chunk['timestamp'][1] - chunk['timestamp'][0]
                
                # Prolongation (> 1.0s)
                if duration > 1.0: prolongations += 1
                
                # Block (> 0.5s pause)
                if i > 0:
                    prev_end = chunks[i-1]['timestamp'][1]
                    if prev_end is not None:
                        pause = chunk['timestamp'][0] - prev_end
                        if pause > 0.5: blocks += 1

                # Repetition
                if i > 0:
                    prev = chunks[i-1]['text'].strip().lower().translate(str.maketrans('', '', string.punctuation))
                    curr = word.lower().translate(str.maketrans('', '', string.punctuation))
                    if prev == curr: repetitions += 1

            self.speech_metrics = {
                "delay": response_time,
                "blocks": blocks,
                "reps": repetitions,
                "prolong": prolongations
            }

            # Update UI
            report = f"Delay: {response_time:.2f}s\nBlocks: {blocks}\nReps: {repetitions}\nProlong: {prolongations}"
            # Add text preview to report for debugging
            report += f"\n\nTranscript:\n{text[:100]}..."
            
            self.txt_audio_result.delete(1.0, tk.END)
            self.txt_audio_result.insert(tk.END, report)
            self.btn_audio_rec.config(state="normal")
            self.check_final_risk()

        except Exception as e:
            print(e)
            self.status_var.set("Error in Audio Analysis")
            self.btn_audio_rec.config(state="normal")

    # --- PART 3: FINAL PREDICTION ALGORITHM ---
    def check_final_risk(self):
        # Only predict if we have data from both steps (or at least one for testing)
        risk_score = 0
        reasons = []

        # 1. Visual Rules
        if self.gaze_score > 0 and self.gaze_score < 50.0:
            risk_score += 2
            reasons.append("Low Eye Contact")

        # 2. Audio Rules
        if self.speech_metrics:
            if self.speech_metrics['delay'] > 2.5:
                risk_score += 1
                reasons.append("Speech Delay")
            if self.speech_metrics['reps'] >= 2:
                risk_score += 1
                reasons.append("Stuttering")
            if self.speech_metrics['blocks'] >= 2:
                risk_score += 1
                reasons.append("Speech Blocks")

        # Display Result
        if risk_score >= 3:
            self.lbl_final_risk.config(text="HIGH RISK OF ASD", fg="#ff4757")
            self.status_var.set(f"High Risk: {', '.join(reasons)}")
        elif risk_score >= 1:
            self.lbl_final_risk.config(text="MODERATE RISK", fg="#eccc68")
            self.status_var.set(f"Moderate Risk: {', '.join(reasons)}")
        else:
            self.lbl_final_risk.config(text="LOW RISK / TYPICAL", fg="#2ed573")
            self.status_var.set("Analysis complete. Behavior appears neurotypical.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrishtiApp(root)
    root.mainloop()