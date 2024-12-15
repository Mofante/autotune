from functools import partial
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import scipy.signal as sig
import psola


SEMITONES_IN_OCTAVE = 12

from my_pyin import my_pyin

def correct_pitch_to_nearest(f0):
    """
    Round the given pitch values to the nearest MIDI note numbers.
    
    Args:
        f0 (np.ndarray): Input pitch values in Hz
    
    Returns:
        np.ndarray: Pitch values rounded to nearest MIDI notes
    """
    # Round to nearest MIDI note
    midi_note = np.around(librosa.hz_to_midi(f0))
    
    # Preserve nan values
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    
    # Convert back to Hz
    return librosa.midi_to_hz(midi_note)


def correct_pitch_to_scale(f0, scale):
    """
    Return the pitch closest to f0 that belongs to the given scale.
    
    Args:
        f0 (float): Input pitch value in Hz
        scale (str): Musical scale
    
    Returns:
        float: Pitch value closest to the input that belongs to the scale
    """
    # Preserve nan
    if np.isnan(f0):
        return np.nan
    
    # Get scale degrees
    degrees = librosa.key_to_degrees(scale)
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    
    # Convert input to MIDI note
    midi_note = librosa.hz_to_midi(f0)
    
    # Get pitch class (remainder when divided by 12)
    degree = midi_note % SEMITONES_IN_OCTAVE
    
    # Find closest pitch class from the scale
    degree_id = np.argmin(np.abs(degrees - degree))
    
    # Calculate difference between input and closest scale pitch
    degree_difference = degree - degrees[degree_id]
    
    # Adjust MIDI note to match scale
    midi_note -= degree_difference
    
    # Convert back to Hz
    return librosa.midi_to_hz(midi_note)


def correct_pitch_array_to_scale(f0, scale):
    """
    Map each pitch in the f0 array to the closest pitch belonging to the given scale.
    
    Args:
        f0 (np.ndarray): Input pitch array in Hz
        scale (str): Musical scale
    
    Returns:
        np.ndarray: Array of pitches corrected to the scale
    """
    # Apply correction to each pitch
    sanitized_pitch = np.zeros_like(f0)
    for i in range(f0.shape[0]):
        sanitized_pitch[i] = correct_pitch_to_scale(f0[i], scale)
    
    # Median filtering to smooth the corrected pitch
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    
    # Restore any NaN values lost during filtering
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    
    return smoothed_sanitized_pitch


def robot_voice_effect(audio, sr, intensity=0.5):
    """
    Apply a robot voice effect by quantizing the audio's amplitude.
    
    Args:
    audio (numpy.ndarray): Input audio signal
    sr (int): Sample rate
    intensity (float): Effect intensity (0.0 to 1.0)
    
    Returns:
    numpy.ndarray: Audio with robot voice effect
    """
    # Clip intensity between 0 and 1
    intensity = np.clip(intensity, 0.0, 1.0)
    
    # Create a quantization effect
    quantization_levels = int(np.round(2 + intensity * 14))  # 2 to 16 levels
    
    # Normalize audio
    audio_norm = audio / np.max(np.abs(audio))
    
    # Quantize the audio
    quantized = np.round(audio_norm * (quantization_levels - 1)) / (quantization_levels - 1)
    
    # Blend original and quantized audio
    return audio * (1 - intensity) + quantized * intensity


def chorus_effect(audio, sr, intensity=0.5):
    """
    Apply a chorus effect by adding slightly pitch-shifted and delayed copies of the audio.
    
    Args:
    audio (numpy.ndarray): Input audio signal
    sr (int): Sample rate
    intensity (float): Effect intensity (0.0 to 1.0)
    
    Returns:
    numpy.ndarray: Audio with chorus effect
    """
    # Clip intensity between 0 and 1
    intensity = np.clip(intensity, 0.0, 1.0)
    
    # Create multiple delayed and pitch-shifted copies
    delays = [0.01, 0.02, 0.03]  # seconds
    pitch_shifts = [0.5, -0.5, 1.0]  # semitones
    
    chorus_audio = audio.copy()
    
    for delay, shift in zip(delays, pitch_shifts):
        # Create delayed version
        delay_samples = int(delay * sr)
        delayed = np.pad(audio, (delay_samples, 0))[:-delay_samples]
        
        # Pitch shift
        shifted = librosa.effects.pitch_shift(delayed, sr=sr, n_steps=shift)
        
        # Mix the delayed and shifted audio
        chorus_audio += shifted * (intensity / len(delays))
    
    # Normalize to prevent clipping
    return chorus_audio / np.max(np.abs(chorus_audio))


def distortion_effect(audio, sr, intensity=0.5):
    """
    Apply a distortion effect by hard clipping the audio signal.
    
    Args:
    audio (numpy.ndarray): Input audio signal
    sr (int): Sample rate
    intensity (float): Effect intensity (0.0 to 1.0)
    
    Returns:
    numpy.ndarray: Audio with distortion effect
    """
    # Clip intensity between 0 and 1
    intensity = np.clip(intensity, 0.0, 1.0)
    
    # Normalize audio
    audio_norm = audio / np.max(np.abs(audio))
    
    # Create hard clipping
    clipping_threshold = 1 - (intensity * 0.9)
    distorted = np.clip(audio_norm / clipping_threshold, -1, 1) * clipping_threshold
    
    # Blend original and distorted audio
    return audio * (1 - intensity) + distorted * intensity


def tremolo_effect(audio, sr, intensity=0.5):
    """
    Apply a tremolo effect by amplitude modulation.
    
    Args:
    audio (numpy.ndarray): Input audio signal
    sr (int): Sample rate
    intensity (float): Effect intensity (0.0 to 1.0)
    
    Returns:
    numpy.ndarray: Audio with tremolo effect
    """
    # Clip intensity between 0 and 1
    intensity = np.clip(intensity, 0.0, 1.0)
    
    # Create a sinusoidal modulation signal
    # Frequency range: 4-7 Hz is typical for tremolo
    mod_freq = 4 + (intensity * 3)
    t = np.linspace(0, len(audio)/sr, len(audio), endpoint=False)
    
    # Modulation signal with depth controlled by intensity
    modulation = 1 - (intensity * 0.5 * np.sin(2 * np.pi * mod_freq * t))
    
    # Apply modulation to audio
    tremolo_audio = audio * modulation
    
    return tremolo_audio


def reverb_effect(audio, sr, intensity=0.5):
    """
    Apply a simple reverb effect using a feedback delay network.
    
    Args:
    audio (numpy.ndarray): Input audio signal
    sr (int): Sample rate
    intensity (float): Effect intensity (0.0 to 1.0)
    
    Returns:
    numpy.ndarray: Audio with reverb effect
    """
    # Clip intensity between 0 and 1
    intensity = np.clip(intensity, 0.0, 1.0)
    
    # Create multiple delay lines with different delays
    delay_times = [0.02, 0.04, 0.06, 0.09]  # seconds
    
    # Create initial reverb signal
    reverb_audio = np.zeros_like(audio)
    
    for delay in delay_times:
        # Create delayed version with decaying amplitude
        delay_samples = int(delay * sr)
        delayed = np.pad(audio, (delay_samples, 0))[:-delay_samples]
        
        # Decay factor increases with intensity
        decay = (1 - intensity) * 0.6
        reverb_audio += delayed * (decay ** (delay_times.index(delay) + 1))
    
    # Mix original and reverb audio
    mixed_audio = audio + reverb_audio * intensity
    
    # Normalize to prevent clipping
    return mixed_audio / np.max(np.abs(mixed_audio))


def autotune(audio, sr, correction_function, effect=None, effect_intensity=0.5):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, _, _ = my_pyin(audio,
                       frame_length=frame_length,
                       hop_length=hop_length,
                       sr=sr,
                       fmin=fmin,
                       fmax=fmax)

    # Apply the chosen adjustment strategy to the pitch.
    corrected_f0 = correction_function(f0)

    # Pitch-shifting using the PSOLA algorithm.
    pitch_corrected_y = psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

    # Apply optional audio effect
    if effect == 'robot':
        pitch_corrected_y = robot_voice_effect(pitch_corrected_y, sr, effect_intensity)
    elif effect == 'chorus':
        pitch_corrected_y = chorus_effect(pitch_corrected_y, sr, effect_intensity)
    elif effect == 'distortion':
        pitch_corrected_y = distortion_effect(pitch_corrected_y, sr, effect_intensity)
    elif effect == 'tremolo':
        pitch_corrected_y = tremolo_effect(pitch_corrected_y, sr, effect_intensity)
    elif effect == 'reverb':
        pitch_corrected_y = reverb_effect(pitch_corrected_y, sr, effect_intensity)

    return pitch_corrected_y


def main():
    # Parse the command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument('vocals_file')
    ap.add_argument('--correction-method', '-c', choices=['closest', 'scale'], default='closest')
    ap.add_argument('--scale', '-s', type=str, help='see librosa.key_to_degrees;'
                                                    ' used only for the \"scale\" correction'
                                                    ' method')
    ap.add_argument('--effect', '-e', choices=['robot', 'chorus', 'distortion', 'tremolo', 'reverb'], 
                    help='Apply an audio effect to the pitch-corrected audio')
    ap.add_argument('--effect-intensity', type=float, default=0.5,
                    help='Intensity of the chosen audio effect (0.0 to 1.0). Default is 0.5.')
    args = ap.parse_args()
    
    filepath = Path(args.vocals_file)

    # Load the audio file.
    y, sr = librosa.load(str(filepath), sr=None, mono=False)

    # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
    if y.ndim > 1:
        y = y[0, :]

    # Pick the pitch adjustment strategy according to the arguments.
    correction_function = correct_pitch_to_nearest if args.correction_method == 'closest' else \
        partial(correct_pitch_array_to_scale, scale=args.scale)

    # Perform the auto-tuning with optional effect.
    pitch_corrected_y = autotune(y, sr, correction_function, 
                                 effect=args.effect, 
                                 effect_intensity=args.effect_intensity)

    # Write the corrected audio to an output file.
    filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
    sf.write(str(filepath), pitch_corrected_y, sr)

    
if __name__=='__main__':
    main()
