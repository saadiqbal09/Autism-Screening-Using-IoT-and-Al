import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
from scipy.io.wavfile import write
import torch
from transformers import pipeline
import numpy as np
import threading
import warnings

warnings.filterwarnings("ignore")

# ---------------- BASIC SETTINGS ----------------
AUDIO_FILE = "speech_input.wav"
SAMPLE_RATE = 16000

# Select model based on system capability
if torch.cuda.is_available():
    MODEL_NAME = "openai/whisper-large-v3"
else:
    MODEL_NAME = "openai/whisper-small"


class ScreeningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech-based ASD Screening Tool")
        self.root.geometry("560x650")
        self.root.configure(bg="#f5f5f5")

        self.recording = False
        self.audio_data = []
        self.asr_model = None
        self.metrics = {}

        self.create_ui()
        self.update_status("Loading speech model...")
        threading.Thread(target=self.load_model, daemon=True).start()

    # ---------------- USER INTERFACE ----------------
    def create_ui(self):
        tk.Label(
            self.root,
            text="Speech-based ASD Screening Tool",
            font=("Segoe UI", 20, "bold"),
            bg="#f5f5f5"
        ).pack(pady=5)

        tk.Label(
            self.root,
            text="Speech-based ASD Screening Tool",
            font=("Segoe UI", 11),
            bg="#f5f5f5"
        ).pack()

        main = tk.Frame(self.root, bg="white", padx=20, pady=20)
        main.pack(padx=20, pady=15, fill="both", expand=True)

        self.record_btn = tk.Button(
            main,
            text="🎙 Start Recording",
            font=("Segoe UI", 12),
            state="disabled",
            command=self.toggle_recording
        )
        self.record_btn.pack(fill="x", pady=6)

        self.upload_btn = tk.Button(
            main,
            text="📂 Upload Audio (.wav)",
            font=("Segoe UI", 12),
            state="disabled",
            command=self.upload_audio
        )
        self.upload_btn.pack(fill="x", pady=6)

        # Language option
        lang_frame = tk.Frame(main, bg="white")
        lang_frame.pack(fill="x", pady=6)

        tk.Label(lang_frame, text="Language:", bg="white").pack(side="left")

        self.language = tk.StringVar(value="auto")
        self.lang_select = ttk.Combobox(
            lang_frame,
            textvariable=self.language,
            state="readonly",
            values=["auto", "english", "hindi", "marathi"],
            width=15
        )
        self.lang_select.pack(side="left", padx=8)

        self.output_box = tk.Text(main, height=10, font=("Consolas", 11))
        self.output_box.pack(fill="both", pady=10)

        self.result_label = tk.Label(
            main,
            text="Result: Pending",
            font=("Segoe UI", 14, "bold"),
            fg="gray",
            bg="white"
        )
        self.result_label.pack(pady=5)

        # Status bar
        self.status = tk.StringVar(value="Initializing...")
        tk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            bg="#e0e0e0",
            font=("Segoe UI", 9)
        ).pack(fill="x", side="bottom")

    # ---------------- STATUS UPDATE ----------------
    def update_status(self, text):
        self.status.set(text)
        self.root.update_idletasks()

    # ---------------- LOAD MODEL ----------------
    def load_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model=MODEL_NAME,
                device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                return_timestamps="word"
            )
            self.record_btn.config(state="normal")
            self.upload_btn.config(state="normal")
            self.update_status("Model ready for analysis")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------- LANGUAGE HANDLING ----------------
    def get_language_code(self):
        choice = self.language.get()
        if choice == "english":
            return "en"
        if choice == "hindi":
            return "hi"
        if choice == "marathi":
            return "mr"
        return None  # auto detect

    # ---------------- RECORD AUDIO ----------------
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_data = []
            self.record_btn.config(text="⏹ Stop Recording")
            self.update_status("Recording audio...")
            threading.Thread(target=self.record_loop, daemon=True).start()
        else:
            self.recording = False
            self.record_btn.config(text="🎙 Start Recording", state="disabled")
            audio = np.concatenate(self.audio_data, axis=0)
            write(AUDIO_FILE, SAMPLE_RATE, (audio * 32767).astype(np.int16))
            self.update_status("Analyzing audio...")
            threading.Thread(
                target=self.analyze_audio,
                args=(AUDIO_FILE,),
                daemon=True
            ).start()

    def record_loop(self):
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback
        ):
            while self.recording:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    # ---------------- UPLOAD AUDIO ----------------
    def upload_audio(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav")]
        )
        if file_path:
            self.update_status("Analyzing uploaded audio...")
            threading.Thread(
                target=self.analyze_audio,
                args=(file_path,),
                daemon=True
            ).start()

    # ---------------- ANALYSIS LOGIC ----------------
    def analyze_audio(self, path):
        try:
            result = self.asr_model(
                path,
                generate_kwargs={
                    "task": "transcribe",
                    "language": self.get_language_code()
                }
            )

            chunks = result["chunks"]
            text = result["text"]

            delay = 0.0
            if chunks and chunks[0]["timestamp"][0] and chunks[0]["timestamp"][0] > 0.8:
                delay = chunks[0]["timestamp"][0]

            blocks = repetitions = prolongations = 0
            previous_end = None
            recent_words = []

            def clean(word):
                return word.lower().strip().replace(",", "").replace(".", "")

            for c in chunks:
                if None in c["timestamp"]:
                    continue

                start, end = c["timestamp"]
                word = clean(c["text"])
                duration = end - start

                if duration > 1.0:
                    prolongations += 1

                if previous_end is not None and start - previous_end > 0.35:
                    blocks += 1

                if len(recent_words) >= 2 and word in recent_words[-2:]:
                    repetitions += 1

                recent_words.append(word)
                previous_end = end

            self.metrics = {
                "delay": delay,
                "blocks": blocks,
                "reps": repetitions,
                "prolong": prolongations
            }

            report = (
                f"Delay: {delay:.2f}s\n"
                f"Blocks: {blocks}\n"
                f"Repetitions: {repetitions}\n"
                f"Prolongations: {prolongations}\n\n"
                f"Transcript:\n{text[:200]}..."
            )

            if repetitions > 0 or blocks > 0:
                report += "\n\nNote: Speech hesitations observed."

            self.output_box.delete(1.0, tk.END)
            self.output_box.insert(tk.END, report)

            self.evaluate_result()
            self.record_btn.config(state="normal")
            self.update_status("Analysis complete")

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            self.update_status("Analysis failed")

    # ---------------- FINAL RESULT ----------------
    def evaluate_result(self):
        score = 0
        d = self.metrics.get("delay", 0)
        r = self.metrics.get("reps", 0)
        b = self.metrics.get("blocks", 0)
        p = self.metrics.get("prolong", 0)

        if d > 3:
            score += 2
        if r >= 3:
            score += 2
        if b >= 2:
            score += 2
        if p >= 3 and (r >= 2 or b >= 1):
            score += 1

        if score >= 4:
            self.result_label.config(text="HIGH RISK", fg="red")
        elif score >= 2:
            self.result_label.config(text="MODERATE RISK", fg="orange")
        else:
            self.result_label.config(text="LOW RISK / TYPICAL", fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    app = ScreeningApp(root)
    root.mainloop()
