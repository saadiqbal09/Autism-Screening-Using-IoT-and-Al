import tkinter as tk
from tkinter import ttk, messagebox
# Removed CV2, MediaPipe, Pillow as they were for visual analysis
import sounddevice as sd
from scipy.io.wavfile import write
import torch
from transformers import pipeline
import numpy as np
import threading
import time
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
        self.root.title("Drishti AI - Speech ASD Screening")
        self.root.geometry("600x750") # Resized for single modality
        self.root.configure(bg="#1e1e1e")

        # --- State Variables ---
        self.is_recording_audio = False
        self.audio_frames = []
        self.whisper_model = None
        
        # Metrics Storage
        self.speech_metrics = {}

        # --- Layout ---
        self.setup_ui()
        
        # --- Load AI Models in Background ---
        self.status_var.set("⏳ Loading AI Model... (This takes a moment)")
        threading.Thread(target=self.load_models, daemon=True).start()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#00d2d3", height=60)
        header.pack(fill="x")
        tk.Label(header, text="DRISHTI AI", font=("Segoe UI", 24, "bold"), bg="#00d2d3", fg="white").pack(pady=5)
        tk.Label(header, text="Voice & Speech Pattern Analysis", font=("Segoe UI", 12), bg="#00d2d3", fg="white").pack(pady=0)

        # Main Layout
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # --- Controls Panel ---
        control_panel = tk.Frame(main_frame, bg="#2d3436")
        control_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(control_panel, text="Speech Response Analysis", font=("Segoe UI", 16, "bold"), bg="#2d3436", fg="#00d2d3").pack(pady=20)
        
        # Language Select
        tk.Label(control_panel, text="Select Language:", font=("Segoe UI", 12), bg="#2d3436", fg="white").pack()
        self.lang_var = tk.StringVar(value="english")
        self.lang_combo = ttk.Combobox(control_panel, textvariable=self.lang_var, values=["english", "hindi", "marathi", "tamil"], state="readonly", font=("Segoe UI", 11), width=20)
        self.lang_combo.pack(pady=10)

        # Record Button
        self.btn_audio_rec = tk.Button(control_panel, text="🎙️ Start Recording", command=self.toggle_audio_recording, bg="#e17055", fg="white", font=("Segoe UI", 14, "bold"), width=20, height=2, state="disabled")
        self.btn_audio_rec.pack(pady=20)
        
        # Result Box
        self.txt_audio_result = tk.Text(control_panel, height=10, width=50, bg="#1e1e1e", fg="#00d2d3", font=("Consolas", 11))
        self.txt_audio_result.pack(pady=10, padx=10)

        tk.Frame(control_panel, height=2, bg="#636e72").pack(fill="x", pady=20) # Separator

        # Final Prediction
        self.lbl_final_risk = tk.Label(control_panel, text="RISK ASSESSMENT: Pending...", font=("Segoe UI", 16, "bold"), bg="#2d3436", fg="gray")
        self.lbl_final_risk.pack(pady=10)

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(self.root, textvariable=self.status_var, bg="#2d3436", fg="white", anchor="w").pack(side="bottom", fill="x")

    def load_models(self):
        # Load Whisper Only
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = pipeline(
                "automatic-speech-recognition", 
                model=WHISPER_MODEL_ID, 
                device=device, 
                return_timestamps="word"
            )
            self.status_var.set(f"✅ AI Model Loaded ({device}). System Ready.")
            self.btn_audio_rec.config(state="normal")
        except Exception as e:
            self.status_var.set(f"❌ Error loading model: {e}")

    # --- AUDIO LOGIC ---
    def toggle_audio_recording(self):
        if not self.is_recording_audio:
            self.is_recording_audio = True
            self.audio_frames = []
            self.btn_audio_rec.config(text="⏹ Stop & Analyze", bg="#d63031")
            self.lbl_final_risk.config(text="RISK ASSESSMENT: Pending...", fg="gray")
            threading.Thread(target=self.record_loop, daemon=True).start()
        else:
            self.is_recording_audio = False
            self.btn_audio_rec.config(text="🎙️ Start Recording", bg="#e17055", state="disabled")
            self.status_var.set("Processing Audio... Please Wait.")
            
            # Save and Analyze
            if self.audio_frames:
                audio_data = np.concatenate(self.audio_frames, axis=0)
                write(AUDIO_FILENAME, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
                threading.Thread(target=self.analyze_audio, daemon=True).start()
            else:
                self.btn_audio_rec.config(state="normal")

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

    # --- FINAL PREDICTION ALGORITHM ---
    def check_final_risk(self):
        # Risk Assessment based purely on Speech Disfluencies
        risk_score = 0
        reasons = []

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
            if self.speech_metrics['prolong'] >= 1:
                risk_score += 1
                reasons.append("Prolongations")

        # Display Result
        if risk_score >= 3:
            self.lbl_final_risk.config(text="HIGH RISK OF ASD", fg="#ff4757")
            self.status_var.set(f"High Risk: {', '.join(reasons)}")
        elif risk_score >= 1:
            self.lbl_final_risk.config(text="MODERATE RISK", fg="#eccc68")
            self.status_var.set(f"Moderate Risk: {', '.join(reasons)}")
        else:
            self.lbl_final_risk.config(text="LOW RISK / TYPICAL", fg="#2ed573")
            self.status_var.set("Analysis complete. Speech patterns appear neurotypical.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrishtiApp(root)
    root.mainloop()
    