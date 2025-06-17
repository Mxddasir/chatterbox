import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def generate_speech(text, output_file, exaggeration=0.5, cfg_weight=0.5):
    # Initialize the model
    model = ChatterboxTTS.from_pretrained(device="cpu")
    
    # Generate speech with custom parameters
    wav = model.generate(
        text,
        exaggeration=exaggeration,  # Controls emotion intensity (0.0 to 1.0)
        cfg_weight=cfg_weight       # Controls how closely to follow the prompt (0.0 to 1.0)
    )
    
    # Save the audio file
    ta.save(output_file, wav, model.sr)
    print(f"Generated {output_file} with exaggeration={exaggeration}, cfg_weight={cfg_weight}")

# Example 1: Normal speech
text1 = "Welcome to Chatterbox TTS! This is a demonstration of normal speech."
generate_speech(text1, "normal_speech.wav")

# Example 2: More expressive speech
text2 = "Wow! This is amazing! I'm so excited to show you what I can do!"
generate_speech(text2, "expressive_speech.wav", exaggeration=0.7, cfg_weight=0.3)

# Example 3: Very dramatic speech
text3 = "In a world where technology reigns supreme, one voice stands above all others!"
generate_speech(text3, "dramatic_speech.wav", exaggeration=0.9, cfg_weight=0.2)

print("\nAll audio files have been generated! Try playing them to hear the differences.") 