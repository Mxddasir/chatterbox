import os
import tempfile
import json
import torchaudio as ta
from flask import Flask, request, send_file, render_template_string, jsonify
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import torch
from pydub import AudioSegment

app = Flask(__name__)
model = ChatterboxTTS.from_pretrained(device="cpu")

# Create necessary directories
VOICES_DIR = Path("voices")
OUTPUT_DIR = Path("output")
VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Store voice metadata
VOICES_METADATA_FILE = VOICES_DIR / "voices.json"

def load_voices_metadata():
    if VOICES_METADATA_FILE.exists():
        with open(VOICES_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_voices_metadata(metadata):
    with open(VOICES_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def merge_audio_clips(clips, sample_rate, silence_duration=0.3):
    """Merge multiple audio clips with natural pauses and fade effects."""
    try:
        # Convert numpy arrays to AudioSegment objects
        audio_segments = []
        for clip in clips:
            # Convert to mono if stereo
            if len(clip.shape) > 1:
                clip = clip.mean(axis=0)
            
            # Convert to int16 format
            clip = (clip * 32767).astype(np.int16)
            
            # Create AudioSegment from numpy array
            audio_segment = AudioSegment(
                clip.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit audio
                channels=1  # mono
            )
            
            # Add fade in/out to each clip (100ms fade)
            audio_segment = audio_segment.fade_in(100).fade_out(100)
            audio_segments.append(audio_segment)
        
        # Create natural pauses with varying durations
        # Shorter pause for quick responses, longer for more thoughtful ones
        pause_durations = [
            int(0.2 * 1000),  # 200ms for quick responses
            int(0.4 * 1000),  # 400ms for normal pauses
            int(0.6 * 1000),  # 600ms for longer pauses
        ]
        
        # Merge clips with natural pauses
        merged = audio_segments[0]
        for i, segment in enumerate(audio_segments[1:], 1):
            # Choose pause duration based on position
            if i < len(audio_segments) - 1:
                pause_duration = pause_durations[i % len(pause_durations)]
            else:
                # Longer pause before the last clip
                pause_duration = int(0.8 * 1000)  # 800ms
            
            # Create pause with slight volume variation
            pause = AudioSegment.silent(duration=pause_duration)
            # Add slight volume variation to make it more natural
            pause = pause - 3  # Reduce volume by 3dB
            
            # Add crossfade between clips (50ms)
            merged = merged.append(pause, crossfade=50)
            merged = merged.append(segment, crossfade=50)
        
        # Add a final fade out to the entire conversation
        merged = merged.fade_out(200)  # 200ms fade out
        
        # Convert back to numpy array
        merged_array = np.array(merged.get_array_of_samples(), dtype=np.float32) / 32767.0
        return merged_array.reshape(1, -1)  # Return as mono 2D array
        
    except Exception as e:
        print(f"Error in merge_audio_clips: {str(e)}")
        raise

# HTML template for the upload form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voice Cloning TTS</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary-color: #2196F3;
            --success-color: #4CAF50;
            --error-color: #f44336;
            --text-color: #333;
            --border-radius: 8px;
            --transition: all 0.3s ease;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: var(--text-color);
            line-height: 1.6;
        }

        .container {
            background-color: white;
            padding: 30px;
            border-radius: var(--border-radius);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        h1, h2 {
            color: var(--primary-color);
            margin-bottom: 20px;
        }

        h1 {
            text-align: center;
            font-size: 2.5em;
        }

        .section {
            margin-bottom: 40px;
            padding: 20px;
            background: #fff;
            border-radius: var(--border-radius);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-color);
            font-weight: 500;
        }

        input[type="text"], input[type="file"], textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: var(--border-radius);
            box-sizing: border-box;
            transition: var(--transition);
            font-size: 16px;
        }

        input[type="text"]:focus, textarea:focus, select:focus {
            border-color: var(--primary-color);
            outline: none;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }

        .voice-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .voice-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: var(--border-radius);
            border: 2px solid #ddd;
            transition: var(--transition);
            margin-bottom: 15px;
        }

        .voice-card:hover {
            border-color: var(--primary-color);
        }

        .voice-card h3 {
            margin: 0 0 10px 0;
            color: var(--primary-color);
        }

        .voice-card audio {
            width: 100%;
            margin: 10px 0;
        }

        .voice-card .actions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        .voice-card button {
            flex: 1;
            padding: 8px;
            font-size: 14px;
            background-color: var(--error-color);
        }

        .script-entry {
            margin-bottom: 15px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: var(--border-radius);
            border: 2px solid #ddd;
            transition: var(--transition);
        }

        .script-entry:hover {
            border-color: var(--primary-color);
        }

        .script-entry select {
            margin-bottom: 15px;
            background-color: white;
        }

        .script-entry textarea {
            min-height: 80px;
            margin-bottom: 10px;
            background-color: white;
        }

        .add-script-btn {
            background-color: var(--success-color);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .add-script-btn::before {
            content: "+";
            font-size: 20px;
        }

        .remove-script-btn {
            background-color: var(--error-color);
            padding: 8px 15px;
            margin-top: 10px;
            width: 100%;
        }

        .progress-container {
            display: none;
            margin: 20px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: var(--border-radius);
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .progress {
            width: 0%;
            height: 100%;
            background-color: var(--primary-color);
            transition: width 0.3s ease;
        }

        .progress-text {
            text-align: center;
            color: #666;
            font-size: 14px;
        }

        .audio-clips {
            margin-top: 30px;
        }

        .audio-clip {
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: var(--border-radius);
            border: 2px solid #ddd;
        }

        .audio-clip h4 {
            margin: 0 0 15px 0;
            color: var(--primary-color);
        }

        .audio-clip audio {
            width: 100%;
            margin: 10px 0;
        }

        .download-info {
            margin-top: 15px;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: var(--border-radius);
            font-size: 14px;
            color: #1976D2;
        }

        .download-btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: var(--primary-color);
            color: white;
            text-decoration: none;
            border-radius: var(--border-radius);
            transition: var(--transition);
        }

        .download-btn:hover {
            background-color: #1976D2;
        }

        .success {
            color: var(--success-color);
            padding: 15px;
            background: #e8f5e9;
            border-radius: var(--border-radius);
            margin: 20px 0;
        }

        .error {
            color: var(--error-color);
            padding: 15px;
            background: #ffebee;
            border-radius: var(--border-radius);
            margin: 20px 0;
        }

        .help-text {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }

        select, textarea, input[type="text"], input[type="file"] {
            font-size: 16px;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: var(--border-radius);
            width: 100%;
            box-sizing: border-box;
            transition: var(--transition);
            background-color: white;
        }

        select:focus, textarea:focus, input[type="text"]:focus {
            border-color: var(--primary-color);
            outline: none;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }

        button {
            font-size: 16px;
            padding: 14px 28px;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            width: 100%;
            font-weight: 500;
            transition: var(--transition);
            background-color: var(--primary-color);
            color: white;
        }

        button:hover {
            background-color: #1976D2;
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(1px);
        }

        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
            transform: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Cloning TTS</h1>
        
        <!-- Voice Upload Section -->
        <div class="section">
            <h2>Upload Voice</h2>
            <form id="voiceForm">
                <div class="form-group">
                    <label for="voiceName">Voice Name:</label>
                    <input type="text" id="voiceName" name="voiceName" required 
                           placeholder="Enter a name for this voice (e.g., Peter, Stewie)">
                    <div class="help-text">Give this voice a unique name that you'll use in conversations</div>
                </div>
                
                <div class="form-group">
                    <label for="voiceAudio">Voice Sample:</label>
                    <input type="file" id="voiceAudio" name="voiceAudio" accept=".wav,.mp3" required>
                    <div class="help-text">Upload a WAV or MP3 file containing the voice you want to clone</div>
                </div>
                
                <button type="submit">Upload Voice</button>
            </form>
        </div>
        
        <!-- Voice List Section -->
        <div class="section">
            <h2>Available Voices</h2>
            <div id="voiceList" class="voice-list">
                <!-- Voices will be populated here -->
            </div>
        </div>
        
        <!-- Script Generation Section -->
        <div class="section">
            <h2>Generate Conversation</h2>
            <form id="scriptForm">
                <div id="scriptEntries">
                    <div class="script-entry">
                        <select name="voices[]" required>
                            <option value="">Select Voice</option>
                        </select>
                        <textarea name="scripts[]" placeholder="Enter what this voice should say..." required></textarea>
                        <button type="button" class="remove-script-btn" onclick="this.parentElement.remove()">Remove Script</button>
                    </div>
                </div>
                
                <button type="button" class="add-script-btn" onclick="addScriptEntry()">Add Another Script</button>
                
                <div class="slider-container">
                    <label for="exaggeration">Exaggeration:</label>
                    <input type="range" id="exaggeration" name="exaggeration" min="0" max="1" step="0.1" value="0.5">
                    <span class="slider-value">0.5</span>
                    <div class="help-text">Adjust how much to exaggerate the voice characteristics</div>
                </div>
                
                <div class="slider-container">
                    <label for="temperature">Temperature:</label>
                    <input type="range" id="temperature" name="temperature" min="0" max="1" step="0.1" value="0.8">
                    <span class="slider-value">0.8</span>
                    <div class="help-text">Control the randomness in generation</div>
                </div>
                
                <div class="slider-container">
                    <label for="cfg_weight">CFG Weight:</label>
                    <input type="range" id="cfg_weight" name="cfg_weight" min="0" max="1" step="0.1" value="0.5">
                    <span class="slider-value">0.5</span>
                    <div class="help-text">Control how closely to follow the reference voice</div>
                </div>
                
                <button type="submit" id="generateButton">Generate Conversation</button>
            </form>
        </div>
        
        <!-- Progress Section -->
        <div id="progressContainer" class="progress-container">
            <div class="progress-bar">
                <div class="progress" id="progress"></div>
            </div>
            <div class="progress-text" id="progressText">Generating...</div>
        </div>
        
        <!-- Result Section -->
        <div id="result"></div>
        
        <!-- Individual Audio Clips -->
        <div id="audioClips" class="audio-clips"></div>
        
        <!-- Merged Audio Preview -->
        <div id="mergedAudio" class="audio-preview" style="display: none;">
            <h3>Complete Conversation</h3>
            <audio controls></audio>
            <div class="download-info">
                <a href="#" download class="download-btn">Download Complete Audio</a>
            </div>
        </div>
    </div>
    
    <script>
        // Update slider values
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            slider.addEventListener('input', () => {
                valueDisplay.textContent = slider.value;
            });
        });

        // Load and display voices
        async function loadVoices() {
            try {
                const response = await fetch('/list_voices');
                const voices = await response.json();
                const voiceList = document.getElementById('voiceList');
                voiceList.innerHTML = '';
                
                if (voices.length === 0) {
                    voiceList.innerHTML = '<p>No voices uploaded yet. Upload a voice to get started!</p>';
                } else {
                    voices.forEach(voice => {
                        const voiceCard = document.createElement('div');
                        voiceCard.className = 'voice-card';
                        voiceCard.innerHTML = `
                            <h3>${voice.name}</h3>
                            <audio controls src="/voice/${voice.name}"></audio>
                            <div class="actions">
                                <button onclick="deleteVoice('${voice.name}')" class="remove-script-btn">Delete</button>
                            </div>
                        `;
                        voiceList.appendChild(voiceCard);
                    });
                }
                
                // Update voice options in all select elements
                const selects = document.querySelectorAll('select[name="voices[]"]');
                selects.forEach(select => {
                    const currentValue = select.value;
                    select.innerHTML = '<option value="">Select Voice</option>';
                    voices.forEach(voice => {
                        const option = document.createElement('option');
                        option.value = voice.name;
                        option.textContent = voice.name;
                        if (voice.name === currentValue) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });
                });
                
                return voices;
            } catch (error) {
                console.error('Error loading voices:', error);
                return [];
            }
        }

        // Delete a voice
        async function deleteVoice(voiceName) {
            if (!confirm(`Are you sure you want to delete the voice "${voiceName}"?`)) {
                return;
            }
            
            try {
                const response = await fetch(`/delete_voice/${voiceName}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    await loadVoices();
                    alert('Voice deleted successfully!');
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        // Handle voice upload
        document.getElementById('voiceForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const submitButton = e.target.querySelector('button[type="submit"]');
            
            try {
                submitButton.disabled = true;
                submitButton.textContent = 'Uploading...';
                
                const response = await fetch('/upload_voice', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    await loadVoices();
                    e.target.reset();
                    alert('Voice uploaded successfully!');
                } else {
                    const error = await response.json();
                    alert(`Error: ${error.error}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            } finally {
                submitButton.disabled = false;
                submitButton.textContent = 'Upload Voice';
            }
        });

        function addScriptEntry() {
            const scriptEntries = document.getElementById('scriptEntries');
            const newEntry = document.createElement('div');
            newEntry.className = 'script-entry';
            newEntry.innerHTML = `
                <select name="voices[]" required>
                    <option value="">Select Voice</option>
                </select>
                <textarea name="scripts[]" placeholder="Enter what this voice should say..." required></textarea>
                <button type="button" class="remove-script-btn" onclick="this.parentElement.remove()">Remove Script</button>
            `;
            scriptEntries.appendChild(newEntry);
            
            // Update voice options in the new select element
            const select = newEntry.querySelector('select');
            const voices = Array.from(document.querySelectorAll('.voice-card h3')).map(h3 => h3.textContent);
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice;
                option.textContent = voice;
                select.appendChild(option);
            });
        }

        // Handle script generation
        document.getElementById('scriptForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const result = document.getElementById('result');
            const audioClips = document.getElementById('audioClips');
            const mergedAudio = document.getElementById('mergedAudio');
            const progressContainer = document.getElementById('progressContainer');
            const generateButton = document.getElementById('generateButton');
            
            // Reset UI
            result.innerHTML = '';
            audioClips.innerHTML = '';
            mergedAudio.style.display = 'none';
            progressContainer.style.display = 'block';
            generateButton.disabled = true;

            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 2;
                if (progress <= 90) {
                    document.getElementById('progress').style.width = `${progress}%`;
                    document.getElementById('progressText').textContent = `Generating... ${progress}%`;
                }
            }, 1000);

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });

                clearInterval(progressInterval);
                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    document.getElementById('progress').style.width = '100%';
                    document.getElementById('progressText').textContent = 'Complete!';
                    
                    // Display individual clips
                    data.clips.forEach((clip, index) => {
                        const clipDiv = document.createElement('div');
                        clipDiv.className = 'audio-clip';
                        clipDiv.innerHTML = `
                            <h4>${clip.voice}: "${clip.script}"</h4>
                            <audio controls src="${clip.audio_url}"></audio>
                            <div class="download-info">
                                <a href="${clip.audio_url}" download>Download Clip</a>
                            </div>
                        `;
                        audioClips.appendChild(clipDiv);
                    });
                    
                    // Display merged audio
                    const mergedAudioElement = mergedAudio.querySelector('audio');
                    mergedAudioElement.src = data.merged_audio_url;
                    mergedAudio.style.display = 'block';
                    
                    // Update download link
                    const downloadLink = mergedAudio.querySelector('.download-btn');
                    downloadLink.href = data.merged_audio_url;
                    
                    result.innerHTML = '<div class="success">Conversation generated successfully!</div>';
                } else {
                    result.innerHTML = `<div class="error">Error: ${data.error || 'Failed to generate conversation'}</div>`;
                }
            } catch (error) {
                result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                progressContainer.style.display = 'none';
                generateButton.disabled = false;
            }
        });

        // Load voices on page load
        document.addEventListener('DOMContentLoaded', () => {
            loadVoices();
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    if 'voiceAudio' not in request.files or 'voiceName' not in request.form:
        return jsonify({"error": "Missing voice audio or name"}), 400
    
    voice_name = request.form['voiceName']
    audio_file = request.files['voiceAudio']
    
    # Save the voice audio
    voice_path = VOICES_DIR / f"{voice_name}.wav"
    audio_file.save(voice_path)
    
    # Update metadata
    metadata = load_voices_metadata()
    metadata[voice_name] = {
        "path": str(voice_path),
        "created_at": str(datetime.now())
    }
    save_voices_metadata(metadata)
    
    return jsonify({"success": True})

@app.route('/list_voices')
def list_voices():
    metadata = load_voices_metadata()
    voices = [{"name": name, "created_at": data["created_at"]} 
              for name, data in metadata.items()]
    return jsonify(voices)

@app.route('/voice/<voice_name>')
def get_voice(voice_name):
    metadata = load_voices_metadata()
    if voice_name not in metadata:
        return jsonify({"error": "Voice not found"}), 404
    
    return send_file(metadata[voice_name]["path"])

@app.route('/delete_voice/<voice_name>', methods=['DELETE'])
def delete_voice(voice_name):
    metadata = load_voices_metadata()
    if voice_name not in metadata:
        return jsonify({"error": "Voice not found"}), 404
    
    # Delete voice file
    voice_path = Path(metadata[voice_name]["path"])
    if voice_path.exists():
        voice_path.unlink()
    
    # Update metadata
    del metadata[voice_name]
    save_voices_metadata(metadata)
    
    return jsonify({"success": True})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get form data
        voices = request.form.getlist('voices[]')
        scripts = request.form.getlist('scripts[]')
        
        if not voices or not scripts or len(voices) != len(scripts):
            return jsonify({"error": "Invalid voice or script data"}), 400
        
        exaggeration = float(request.form.get('exaggeration', 0.5))
        temperature = float(request.form.get('temperature', 0.8))
        cfg_weight = float(request.form.get('cfg_weight', 0.5))
        
        # Get voice paths
        metadata = load_voices_metadata()
        timestamp = int(time.time())
        generated_clips = []
        audio_clips = []
        
        # Generate audio for each script
        for i, (voice_name, script) in enumerate(zip(voices, scripts)):
            if voice_name not in metadata:
                return jsonify({"error": f"Voice not found: {voice_name}"}), 404
            
            voice_path = metadata[voice_name]["path"]
            print(f"Generating audio {i+1}/{len(scripts)} for voice: {voice_name}")
            print(f"Script: {script}")
            
            # Generate audio
            wav = model.generate(
                script,
                audio_prompt_path=voice_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight
            )
            
            # Save individual clip
            clip_filename = f"{voice_name}_{timestamp}_{i}.wav"
            clip_path = OUTPUT_DIR / clip_filename
            ta.save(clip_path, wav, model.sr)
            print(f"Saved clip to: {clip_path}")
            
            # Store clip info and audio data
            generated_clips.append({
                "voice": voice_name,
                "script": script,
                "audio_url": f"/audio/{clip_filename}"
            })
            audio_clips.append(wav.numpy())
        
        # Merge all clips
        print("Merging audio clips...")
        merged_audio = merge_audio_clips(audio_clips, model.sr)
        
        # Save merged audio
        merged_filename = f"conversation_{timestamp}.wav"
        merged_path = OUTPUT_DIR / merged_filename
        ta.save(merged_path, torch.from_numpy(merged_audio), model.sr)
        print(f"Saved merged audio to: {merged_path}")
        
        return jsonify({
            "clips": generated_clips,
            "merged_audio_url": f"/audio/{merged_filename}",
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error during conversation generation: {str(e)}")
        return jsonify({
            "error": f"Conversation generation failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(OUTPUT_DIR / filename, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 