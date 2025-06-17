import os
import tempfile
import torchaudio as ta
from flask import Flask, request, send_file, render_template_string, jsonify
from chatterbox.tts import ChatterboxTTS

app = Flask(__name__)
model = ChatterboxTTS.from_pretrained(device="cpu")

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

        h1 {
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
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

        input[type="text"], input[type="file"], input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: var(--border-radius);
            box-sizing: border-box;
            transition: var(--transition);
        }

        input[type="text"]:focus, input[type="number"]:focus {
            border-color: var(--primary-color);
            outline: none;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }

        .slider-container {
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: var(--border-radius);
        }

        .slider-container label {
            display: inline-block;
            width: 150px;
            margin-bottom: 0;
        }

        .slider-container input[type="range"] {
            width: 200px;
            height: 6px;
            -webkit-appearance: none;
            background: #ddd;
            border-radius: 3px;
            outline: none;
        }

        .slider-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
            transition: var(--transition);
        }

        .slider-container input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1);
        }

        .slider-value {
            display: inline-block;
            width: 50px;
            text-align: right;
            font-weight: 500;
            color: var(--primary-color);
        }

        button {
            background-color: var(--primary-color);
            color: white;
            padding: 14px 28px;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            width: 100%;
            font-size: 16px;
            font-weight: 500;
            transition: var(--transition);
            margin-bottom: 15px;
        }

        button:hover {
            background-color: #1976D2;
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(1px);
        }

        .progress-container {
            display: none;
            margin: 20px 0;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress {
            width: 0%;
            height: 100%;
            background-color: var(--primary-color);
            transition: width 0.3s ease;
        }

        .progress-text {
            text-align: center;
            margin-top: 10px;
            color: #666;
        }

        .audio-preview {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: var(--border-radius);
            text-align: center;
        }

        .audio-preview h3 {
            color: var(--primary-color);
            margin-bottom: 15px;
        }

        audio {
            width: 100%;
            margin: 15px 0;
        }

        .download-info {
            margin-top: 15px;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: var(--border-radius);
            font-size: 14px;
            color: #1976D2;
        }

        .error {
            color: var(--error-color);
            margin-top: 10px;
            padding: 10px;
            background-color: #ffebee;
            border-radius: var(--border-radius);
        }

        .success {
            color: var(--success-color);
            margin-top: 10px;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: var(--border-radius);
        }

        .help-text {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }

        .preview-container {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: var(--border-radius);
        }

        .preview-container h3 {
            color: var(--primary-color);
            margin-bottom: 15px;
        }

        .preview-audio {
            width: 100%;
            margin: 10px 0;
        }

        .download-button {
            display: inline-block;
            padding: 10px 20px;
            background-color: var(--success-color);
            color: white;
            text-decoration: none;
            border-radius: var(--border-radius);
            transition: var(--transition);
        }

        .download-button:hover {
            background-color: #388E3C;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
            animation: pulse 2s infinite;
        }

        .loading:after {
            content: "Generating speech";
            animation: dots 1.5s steps(5, end) infinite;
        }

        @keyframes dots {
            0%, 20% { content: "Generating speech ."; }
            40% { content: "Generating speech .."; }
            60% { content: "Generating speech ..."; }
            80%, 100% { content: "Generating speech ...."; }
        }

        /* Responsive Design */
        @media (max-width: 600px) {
            .container {
                padding: 15px;
            }

            .slider-container label {
                width: 120px;
            }

            .slider-container input[type="range"] {
                width: 150px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Cloning TTS</h1>
        <form id="ttsForm" action="/generate" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="text">Text to Synthesize:</label>
                <input type="text" id="text" name="text" required 
                       placeholder="Enter text to convert to speech..."
                       aria-label="Text to synthesize">
                <div class="help-text">Enter the text you want to convert to speech</div>
            </div>
            
            <div class="form-group">
                <label for="audio">Reference Audio:</label>
                <input type="file" id="audio" name="audio" accept=".wav,.mp3" required
                       aria-label="Reference audio file">
                <div class="help-text">Upload a WAV or MP3 file containing the voice you want to clone</div>
            </div>

            <div class="slider-container">
                <label for="exaggeration">Exaggeration:</label>
                <input type="range" id="exaggeration" name="exaggeration" 
                       min="0.25" max="2.0" step="0.05" value="0.5"
                       aria-label="Exaggeration level">
                <span class="slider-value">0.5</span>
                <div class="help-text">Adjust how much to exaggerate the voice characteristics (0.25-2.0)</div>
            </div>

            <div class="slider-container">
                <label for="temperature">Temperature:</label>
                <input type="range" id="temperature" name="temperature" 
                       min="0.05" max="5.0" step="0.05" value="0.8"
                       aria-label="Temperature level">
                <span class="slider-value">0.8</span>
                <div class="help-text">Control the randomness in generation (0.05-5.0)</div>
            </div>

            <div class="slider-container">
                <label for="cfg_weight">CFG Weight:</label>
                <input type="range" id="cfg_weight" name="cfg_weight" 
                       min="0.0" max="1.0" step="0.05" value="0.5"
                       aria-label="CFG weight">
                <span class="slider-value">0.5</span>
                <div class="help-text">Control how closely to follow the reference voice (0.0-1.0)</div>
            </div>

            <div class="form-group">
                <label for="seed">Seed:</label>
                <input type="number" id="seed" name="seed" value="0" min="0"
                       aria-label="Random seed">
                <div class="help-text">Set to 0 for random generation, or use a specific number for reproducible results</div>
            </div>

            <button type="submit" id="generateButton">Generate Speech</button>
        </form>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress" id="progressBar"></div>
            </div>
            <div class="progress-text" id="progressText">Initializing...</div>
        </div>

        <div class="loading" id="loading"></div>
        <div id="result"></div>
        <div class="preview-container" id="previewContainer">
            <h3>Generated Audio Preview</h3>
            <audio controls class="preview-audio" id="previewAudio">
                Your browser does not support the audio element.
            </audio>
            <a href="#" class="download-button" id="downloadButton">Download Audio</a>
        </div>
        <div class="download-info" id="downloadInfo"></div>
    </div>

    <script>
        // Update slider values in real-time
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            slider.addEventListener('input', () => {
                valueDisplay.textContent = slider.value;
            });
        });

        // Simulate progress updates
        function updateProgress(progress) {
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            progressBar.style.width = `${progress}%`;
            
            if (progress < 30) {
                progressText.textContent = 'Loading model...';
            } else if (progress < 60) {
                progressText.textContent = 'Processing audio...';
            } else if (progress < 90) {
                progressText.textContent = 'Generating speech...';
            } else {
                progressText.textContent = 'Finalizing...';
            }
        }

        // Handle form submission
        document.getElementById('ttsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const form = e.target;
            const formData = new FormData(form);
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const previewContainer = document.getElementById('previewContainer');
            const downloadInfo = document.getElementById('downloadInfo');
            const progressContainer = document.getElementById('progressContainer');
            const generateButton = document.getElementById('generateButton');
            
            // Reset UI
            loading.style.display = 'block';
            result.innerHTML = '';
            previewContainer.style.display = 'none';
            downloadInfo.innerHTML = '';
            progressContainer.style.display = 'block';
            generateButton.disabled = true;

            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progress <= 100) {
                    updateProgress(progress);
                }
            }, 500);

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });

                clearInterval(progressInterval);
                updateProgress(100);

                if (response.ok) {
                    const blob = await response.blob();
                    const audioUrl = URL.createObjectURL(blob);
                    
                    // Update audio preview
                    const audioElement = document.getElementById('previewAudio');
                    audioElement.src = audioUrl;
                    previewContainer.style.display = 'block';
                    
                    // Update download button
                    const downloadButton = document.getElementById('downloadButton');
                    downloadButton.href = audioUrl;
                    downloadButton.download = 'generated_tts.wav';
                    
                    // Show download info
                    downloadInfo.innerHTML = `
                        <strong>Note:</strong> The audio file will be saved to your Downloads folder.
                        <br>You can preview the audio above before downloading.
                    `;
                    
                    result.innerHTML = '<div class="success">Speech generated successfully!</div>';
                } else {
                    const error = await response.json();
                    result.innerHTML = `<div class="error">Error: ${error.error}</div>`;
                }
            } catch (error) {
                result.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                loading.style.display = 'none';
                progressContainer.style.display = 'none';
                generateButton.disabled = false;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_tts():
    if 'audio' not in request.files or 'text' not in request.form:
        return jsonify({"error": "Missing audio file or text"}), 400
    audio_file = request.files['audio']
    text = request.form['text']
    exaggeration = float(request.form.get('exaggeration', 0.5))
    temperature = float(request.form.get('temperature', 0.8))
    cfg_weight = float(request.form.get('cfg_weight', 0.5))
    seed = int(request.form.get('seed', 0))

    # Save the uploaded audio file temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_file.save(temp_audio.name)
        reference_audio = temp_audio.name

    # Convert MP3 to WAV if needed
    if reference_audio.endswith('.mp3'):
        wav_path = reference_audio.replace('.mp3', '.wav')
        if not os.path.exists(wav_path):
            waveform, sample_rate = ta.load(reference_audio)
            ta.save(wav_path, waveform, sample_rate)
        reference_audio = wav_path

    # Generate TTS
    wav = model.generate(
        text,
        audio_prompt_path=reference_audio,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfg_weight
    )

    # Save the generated audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
        ta.save(temp_output.name, wav, model.sr)
        return send_file(temp_output.name, mimetype='audio/wav', as_attachment=True, download_name='generated_tts.wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 