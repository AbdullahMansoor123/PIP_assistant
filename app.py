
import tkinter as tk
from tkinter import messagebox
import threading
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import soundfile as sf
import whisper
import time
from kokoro import KPipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from ollama import generate

# Mock functions for demonstration — replace with real logic
def generate_job_questions(jd_file_path):
    """
    jd_file: job description file path
    Returns: list of interview questions as strings
    """
    # Read job description from file
    with open(jd_file_path, "r") as file:
        job_description = file.read()

    # Build the prompt for generating interview questions
    prompt = f"""
    You are a helpful assistant trained to generate interview questions based on job descriptions.

    Task: Read the job description provided and generate relevant interview questions that assess a candidate’s fit for the role.

    Instructions:
    - Focus on the required skills, responsibilities, and qualifications.
    - Include a mix of technical, behavioral, and situational questions.
    - Generate between 2 to 3 interview questions.
    - Return ONLY the questions as a clean, numbered list (e.g., 1. ..., 2. ..., etc.).
    - Do NOT include any introduction, explanation, or summary. Only output the questions.

    JOB DESCRIPTION:
    {job_description}
    """

    # Call Ollama using your custom or local LLaMA 3.2:1B model
    response = generate(
        model="llama3.2:1b",  # Replace with your actual model name
        prompt=prompt
    )

    # Extract and split the response into a list of questions
    raw_response = response['response']
    
    # Try splitting by numbered list
    questions = [
        q.strip().lstrip("0123456789. ").strip()
        for q in raw_response.strip().split("\n")
        if q.strip()
    ]

    # Filter out any empty or non-question lines
    questions = [q for q in questions if "?" in q]

    return questions


# Convert User Voice to Text
import whisper

def voice_2_txt(audio_path='user_answer.wav', model_size='base'):
    """
    Transcribes spoken audio from a file using OpenAI's Whisper model.

    Parameters:
    - audio_path (str): Path to the audio file
    - model_size (str): Whisper model size (tiny, base, small, medium, large)

    Returns:
    - candidate_response (str): The transcribed text from the audio
    """
    print(f"Loading Whisper model: '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Converting Voice to Text: '{audio_path}'...")
    result = model.transcribe(audio=audio_path)
    
    candidate_response = result["text"]
    print("Voice to Test Conversion Complete!")
    
    return candidate_response


# Evaluate the User Answer and give feedback
def evaluate_responses(jd_file_path, qa_dict):
    """
    Evaluates a set of candidate answers to multiple interview questions using LLM scoring and feedback.

    Parameters:
    - jd_file_path: Path to the job description file.
    - qa_dict: Dictionary with interview_question as key and (input_method, candidate_response) as value.
               input_method: 'v' for voice (uses voice_2_txt), 't' for text (uses provided response)

    Returns:
    - A single comprehensive evaluation report covering all questions and answers.
    """

    # Read job description from file
    with open(jd_file_path, "r") as file:
        job_description = file.read()

    # Gather all responses
    combined_qna = ""
    for i, (question, (input_method, response)) in enumerate(qa_dict.items(), start=1):
        if input_method == 'v':
            response = voice_2_txt()
        combined_qna += f"\nQuestion {i}: {question}\nAnswer: {response}\n"

    # Prompt for consolidated evaluation
    prompt = f"""
    You are an experienced interview evaluator.

    Task: Assess the candidate's responses to the following interview questions, considering the provided job description.

    Instructions:
    1. For EACH question-response pair:
       - Evaluate based on the following criteria:
         - Relevance
         - Clarity
         - Depth
         - Communication Skills
         - Alignment with Job Requirements
       - For each criterion, provide:
         - A score from 1 (Poor) to 5 (Excellent)
         - A short justification for the score

    2. Then provide:
       - An overall score out of {len(qa_dict) * 25}
       - A summary assessment of the candidate's suitability for the role based on all responses
       - Constructive feedback on how the candidate could improve for future interviews

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE RESPONSES:
    {combined_qna}
    """

    response = generate(
        model="llama3.2:1b",
        prompt=prompt
    )

    return response['response']



# TTS using Kokoro
def ask_question_audio(interview_question, filename="question_audio.wav"):
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    generator = pipeline(text=interview_question, voice='hm_omega')
    for i, (_, _, audio) in enumerate(generator):
        if i == 0:  # Play only first chunk
            sf.write(filename, audio, 24000)
            break



# Constants
TEXT_FILENAME = 'user_answer.txt'
AUDIO_FILENAME = 'user_answer.wav'
SAMPLE_RATE = 44100

def save_report_to_pdf(qa_dict, evaluation_summary, filename="Interview_Evaluation_Report.pdf"):
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="QTitle", fontSize=12, leading=16, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name="Answer", fontSize=10.5, leading=14, spaceAfter=10))
    styles.add(ParagraphStyle(name="Meta", fontSize=9.5, leading=12, textColor=colors.grey))
    styles.add(ParagraphStyle(name="CustomBullet", parent=styles["Normal"], leftIndent=15, bulletIndent=0, spaceBefore=6))
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=14, leading=18, spaceAfter=12, alignment=1))  # centered

    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=36)
    story = []

    story.append(Paragraph("Interview Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 24))

    for idx, (question, (method, answer)) in enumerate(qa_dict.items(), 1):
        # Create a bordered table-style card for each question-answer pair
        data = [
            [Paragraph(f"Q{idx}: {question}", styles["QTitle"])],
            [Paragraph(f"<b>Response Method:</b> {'Voice (Transcribed)' if method == 'v' else 'Text'}", styles["Meta"])],
            [Paragraph(answer, styles["Answer"])]
        ]

        table = Table(data, colWidths=[doc.width], hAlign="LEFT", style=TableStyle([
            ('BOX', (0,0), (-1,-1), 0.5, colors.black),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (0,0), colors.whitesmoke),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))

        story.append(KeepTogether(table))
        story.append(Spacer(1, 12))

        # Insert page break every 5 questions
        if idx % 5 == 0 and idx != len(qa_dict):
            story.append(PageBreak())

    # Page Break before Evaluation
    story.append(PageBreak())

    story.append(Paragraph("Final Evaluation Summary", styles["SectionHeader"]))
    story.append(Spacer(1, 16))

    # ---- Format Evaluation Text ---- #
    def parse_evaluation_text(text):
        lines = text.splitlines()
        blocks = []

        for line in lines:
            if line.startswith("### **Overall Score**"):
                blocks.append(Paragraph("🏅 <b>Overall Score</b>", styles["Heading2"]))
            elif line.startswith("### **Summary Assessment of Suitability**"):
                blocks.append(Paragraph("🧠 <b>Suitability Summary</b>", styles["Heading2"]))
            elif line.startswith("### **Criteria Evaluations**"):
                blocks.append(Spacer(1, 10))
                blocks.append(Paragraph("📊 <b>Criteria Evaluations</b>", styles["Heading2"]))
            elif line.startswith("### **Feedback**"):
                blocks.append(Spacer(1, 12))
                blocks.append(Paragraph("📝 <b>Feedback</b>", styles["Heading2"]))
            elif line.startswith("#### **"):
                blocks.append(Spacer(1, 6))
                blocks.append(Paragraph(line.replace("#### **", "<b>").replace("**", "</b>"), styles["Heading3"]))
            elif line.strip().startswith(("1. ", "2. ", "3. ", "4. ")):
                blocks.append(Paragraph(line.strip(), styles["CustomBullet"]))
            elif line.strip():
                blocks.append(Paragraph(line.strip(), styles["Normal"]))
            else:
                blocks.append(Spacer(1, 6))
        return blocks

    story.extend(parse_evaluation_text(evaluation_summary))

    doc.build(story)
    print(f"✅ PDF Report saved as: {filename}")
    return filename



# Main App
class InterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Interview Assistant")
        self.questions = []
        self.qa_dict = {}
        self.current_index = 0
        self.recording = False
        self.frames = []

        self.setup_ui()
        pygame.mixer.init()

    def setup_ui(self):
        self.frame = tk.Frame(self.root, padx=20, pady=20)
        self.frame.pack(expand=True, fill='both')
        self.status_label = tk.Label(self.frame, text="Click 'Start Interview' to begin.")
        self.status_label.pack(pady=10)
        self.start_button = tk.Button(self.frame, text="Start Interview", command=self.start_interview)
        self.start_button.pack(pady=5)

    def start_interview(self):
        self.start_button.config(state="disabled")
        self.status_label.config(text="Generating questions...")
        threading.Thread(target=self.load_questions).start()

    def load_questions(self):
        self.questions = generate_job_questions("jd.txt")
        self.root.after(0, self.show_question)

    def show_question(self):
        if self.current_index >= len(self.questions):
            self.finish_interview()
            return

        for widget in self.frame.winfo_children():
            widget.destroy()

        question = self.questions[self.current_index]
        tk.Label(self.frame, text=f"Question {self.current_index + 1}:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 5))
        tk.Label(self.frame, text=question, wraplength=400).pack(anchor="w")
        tk.Button(self.frame, text="🔊 Play Question Audio", command=lambda: self.play_audio_for_question(question)).pack(pady=5)

        self.input_method = tk.StringVar(value="text")
        tk.Radiobutton(self.frame, text="Text", variable=self.input_method, value="text", command=self.toggle_input_method).pack(anchor="w")
        tk.Radiobutton(self.frame, text="Voice", variable=self.input_method, value="voice", command=self.toggle_input_method).pack(anchor="w")

        self.text_input = tk.Text(self.frame, height=4, width=50)
        self.text_input.pack(pady=5)

        self.record_frame = tk.Frame(self.frame)
        self.start_record_btn = tk.Button(self.record_frame, text="🎙️ Start Recording", command=self.start_recording)
        self.stop_record_btn = tk.Button(self.record_frame, text="🛑 Stop Recording", command=self.stop_recording)
        self.start_record_btn.pack(side="left", padx=5)
        self.stop_record_btn.pack(side="left", padx=5)
        self.record_frame.pack(pady=5)

        self.submit_button = tk.Button(self.frame, text="Submit Answer", command=self.submit_answer)
        self.submit_button.pack(pady=10)

        self.status_label = tk.Label(self.frame, text="")
        self.status_label.pack(pady=5)

        self.toggle_input_method()

    def toggle_input_method(self):
        method = self.input_method.get()
        self.text_input.config(state="normal" if method == "text" else "disabled")
        self.start_record_btn.config(state="disabled" if method == "text" else "normal")
        self.stop_record_btn.config(state="disabled" if method == "text" else "normal")

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=2, callback=self.audio_callback)
        self.stream.start()
        self.status_label.config(text="Recording... Speak now!")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            audio_data = np.concatenate(self.frames, axis=0)
            write(AUDIO_FILENAME, SAMPLE_RATE, audio_data)
            self.status_label.config(text="Recording stopped. Voice saved.")
            print(f"Audio saved as '{AUDIO_FILENAME}'")

    def audio_callback(self, indata, frames_count, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def play_audio_for_question(self, question):
        audio_path = "temp_question.wav"
        ask_question_audio(question, audio_path)
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

    def submit_answer(self):
        method = self.input_method.get()
        question = self.questions[self.current_index]

        if method == "text":
            answer = self.text_input.get("1.0", tk.END).strip()
            if not answer:
                messagebox.showwarning("Input Needed", "Please enter a text response.")
                return
            with open(TEXT_FILENAME, 'w') as f:
                f.write(answer)

        else:  # voice
            if not os.path.exists(AUDIO_FILENAME):
                messagebox.showwarning("Audio Not Found", "Please record your answer before submitting.")
                return
            try:
                self.status_label.config(text="Transcribing voice to text...")
                self.root.update()
                transcribed_text = voice_2_txt(audio_path=AUDIO_FILENAME, model_size="base")
                answer = transcribed_text
                with open(TEXT_FILENAME, 'w') as f:
                    f.write(answer)
                self.status_label.config(text="Transcription complete.")
            except Exception as e:
                messagebox.showerror("Transcription Failed", f"Error: {str(e)}")
                return

        self.qa_dict[question] = (method, answer)
        self.current_index += 1
        self.show_question()

    def finish_interview(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        tk.Label(self.frame, text="Generating evaluation report...", font=("Arial", 12)).pack(pady=10)
        self.root.after(100, lambda: threading.Thread(target=self.display_report).start())

    def display_report(self):
        report = evaluate_responses("jd.txt", self.qa_dict)
        self.root.after(0, lambda: self.show_report(report))

    def show_report(self, report):
        for widget in self.frame.winfo_children():
            widget.destroy()

        tk.Label(self.frame, text="Evaluation Report", font=("Arial", 14, "bold")).pack(pady=10)
        text_box = tk.Text(self.frame, wrap="word", height=20)
        text_box.insert(tk.END, report)
        text_box.config(state="disabled")
        text_box.pack(expand=True, fill="both")

        def save_pdf():
            filename = save_report_to_pdf(self.qa_dict, report)
            messagebox.showinfo("Saved", f"PDF saved as: {filename}")

        tk.Button(self.frame, text="💾 Save as PDF", command=save_pdf).pack(pady=5)
        tk.Button(self.frame, text="Close", command=self.root.quit).pack(pady=5)


# Launch app
if __name__ == "__main__":
    root = tk.Tk()
    app = InterviewApp(root)
    root.mainloop()
    