import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import os
import uuid

# Load models and processor
@st.cache_resource
def load_models():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder

processor, model, vocoder = load_models()

# Load speaker embeddings
@st.cache_data
def load_speaker_embeddings():
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return speaker_embeddings

speaker_embeddings = load_speaker_embeddings()

st.title("Text-to-Speech App")

# Get user input
text_input = st.text_area("Enter the text you want to convert to speech:", "Hello, my dog is cute.")

if st.button("Generate Speech"):
    if text_input:
        with st.spinner("Generating speech... Please wait."):
            # Process input and generate speech
            inputs = processor(text=text_input, return_tensors="pt")
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

            # Save the audio file with a unique filename
            output_filename = f"speech_{uuid.uuid4()}.wav"
            sf.write(output_filename, speech.numpy(), samplerate=16000)

        st.success("Speech generated successfully!")

        # Display audio player and download button
        st.audio(output_filename)
        
        with open(output_filename, "rb") as file:
            btn = st.download_button(
                label="Download Audio",
                data=file,
                file_name=output_filename,
                mime="audio/wav"
            )
        
        # Clean up the generated file
        os.remove(output_filename)
    else:
        st.warning("Please enter some text before generating speech.")
