import os
import uuid
from flask import Flask, render_template, request, send_file
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename

import autotune

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        # If file is allowed
        if file and allowed_file(file.filename):
            # Generate a unique filename
            unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(input_filepath)

            # Get correction method and scale
            correction_method = request.form.get('correction_method', 'closest')
            scale = request.form.get('scale', '')
            effect = request.form.get('effect', '')
            effect_intensity = float(request.form.get('effect_intensity', 0.5))

            # Load the audio file
            y, sr = librosa.load(input_filepath, sr=None, mono=False)

            # Handle stereo files
            if y.ndim > 1:
                y = y[0, :]

            # Choose correction function
            if correction_method == 'closest':
                correction_function = autotune.correct_pitch_to_nearest
            else:
                from functools import partial
                correction_function = partial(autotune.correct_pitch_array_to_scale, scale=scale)

            # Perform pitch correction with optional effect
            pitch_corrected_y = autotune.autotune(
                y, sr, 
                correction_function, 
                effect=effect, 
                effect_intensity=effect_intensity
            )

            # Generate output filename
            output_filename = unique_filename.rsplit('.', 1)[0] + '_pitch_corrected.wav'
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            # Save the corrected audio
            sf.write(output_filepath, pitch_corrected_y, sr)

            # Return the page with the processed file
            return render_template('index.html', 
                                   output_file=output_filename, 
                                   input_filename=file.filename)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
