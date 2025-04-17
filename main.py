from ollama import generate


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


# Play question

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf

def ask_questions(interview_question):
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    generator = pipeline(text= interview_question, voice='hm_omega') # hm_omega
    for i, (gs, ps, audio) in enumerate(generator):
        # print(i, gs, ps)
        print(gs)
        display(Audio(data=audio, rate=24000, autoplay=i==0))
        # sf.write(f'{i}.wav', audio, 24000) #UNCOMMENT TO SAVE AUDIO FILES


# Get User Answer in Text/Voice

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading

def get_candidate_response(sample_rate=44100,
                           text_filename='user_answer.txt' ,
                           audio_filename='user_answer.wav'):
    """
    Gets candidate's response either via text or voice input.

    Parameters:
    - input_type (str): 'manual' or 'voice'
    - duration (int): Duration of audio recording in seconds (used for voice input)
    - sample_rate (int): Sampling rate for the recording
    - text_filename (str): Filename to save the manual input text
    - audio_filename (str): Filename to save the voice recording

    Returns:
    - If manual input: returns the input_type and input string.
    - If voice input: returns the input_type and audio file path.
    """
    
    input_type = input("Select Input Method: 'v' for Voice or 't' for Text: ")
    
    if input_type == 't':
        response = input("Please enter your Anwser: ")

        # Save text response to file
        with open(text_filename, 'w') as f:
            f.write(response)
        
        print(f"User Answer saved to '{text_filename}'")
        return (input_type, response)
        
    
    elif input_type == 'v':
        print("Press Enter to START recording...")
        input()
        print("Recording... Press Enter again to STOP.")

        frames = []
        recording = True

        def callback(indata, frames_count, time_info, status):
            if status:
                print(f"Status: {status}")
            if recording:
                frames.append(indata.copy())

        # Use a stream to allow dynamic control
        stream = sd.InputStream(samplerate=sample_rate, channels=2, callback=callback)
        stream.start()

        # Wait for user to press Enter to stop
        input()
        recording = False
        stream.stop()
        stream.close()

        # Concatenate and save the recorded audio
        audio_data = np.concatenate(frames, axis=0)
        write(audio_filename, sample_rate, audio_data)
        print(f"Recording stopped. Audio saved as '{audio_filename}'")

        return (input_type, audio_filename)

    
    else:
        raise ValueError("Invalid input_type. Use 'v' for Voice or 't' for Text")
    

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



def main():
    #1. Generate Question using Job description
    job_questions = generate_job_questions(jd_file_path= "jd.txt")

    #2. Ask Question from User
    import warnings
    warnings.filterwarnings("ignore")
    
    interview_record = {}
    for question in job_questions:
        print(question)
        # show question and option for show/play question in audio
        # ask_questions(question)

        #3. Take user response to question in either voice or text
        (input_method, user_answer) = get_candidate_response()

        #4. Evaluate the User response with respect to question and job discription
        interview_record[question] = (input_method, user_answer)


    eval_report = evaluate_responses(jd_file_path = "jd.txt", 
                                    qa_dict = interview_record)
    print(eval_report)


if __name__ == "__main__":
    main()