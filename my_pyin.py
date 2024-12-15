import numpy as np
import scipy.signal as sig

def my_pyin(audio, frame_length, hop_length, sr, fmin, fmax, threshold=0.1, beta=1.0):
    """
    Custom implementation of the PYIN (Probabilistic YIN) pitch estimation algorithm.
    
    Args:
    audio (numpy.ndarray): Input audio signal
    frame_length (int): Length of each analysis frame
    hop_length (int): Number of samples between successive frames
    sr (int): Sample rate
    fmin (float): Minimum frequency to consider
    fmax (float): Maximum frequency to consider
    threshold (float): Threshold for voiced/unvoiced detection
    beta (float): Parameter to control voiced/unvoiced likelihood
    
    Returns:
    tuple: (f0, voiced_flag, voiced_probabilities)
        - f0: Estimated fundamental frequencies for each frame
        - voiced_flag: Boolean array indicating voiced/unvoiced frames
        - voiced_probabilities: Probability of voicing for each frame
    """
    # Calculate number of frames
    n_frames = 1 + int((len(audio) - frame_length) / hop_length)
    
    # Preallocate arrays
    f0 = np.full(n_frames, np.nan)
    voiced_probabilities = np.zeros(n_frames)
    
    # Frequency range in terms of periods
    period_min = int(sr / fmax)
    period_max = int(sr / fmin)
    
    for frame_idx in range(n_frames):
        # Extract frame
        start = frame_idx * hop_length
        frame = audio[start:start+frame_length]
        
        # Normalize frame
        frame = frame - np.mean(frame)
        
        # Compute difference function
        diff_func = np.zeros(period_max + 1)
        for tau in range(period_max + 1):
            if tau == 0:
                diff_func[tau] = np.sum(frame**2)
            else:
                diff = frame[:len(frame)-tau] - frame[tau:]
                diff_func[tau] = np.sum(diff**2)
        
        # Compute cumulative mean normalized difference function
        cmnd = np.zeros_like(diff_func)
        cmnd[0] = 1.0
        for tau in range(1, len(cmnd)):
            cmnd[tau] = diff_func[tau] / ((1/tau) * np.sum(diff_func[1:tau+1]))
        
        # Find local minima in the specified period range
        local_minima = []
        for tau in range(period_min, period_max):
            if (cmnd[tau-1] > cmnd[tau]) and (cmnd[tau] < cmnd[tau+1]):
                local_minima.append(tau)
        
        # If no local minima found, mark as unvoiced
        if not local_minima:
            f0[frame_idx] = np.nan
            voiced_probabilities[frame_idx] = 0
            continue
        
        # Find global minimum within local minima
        global_min_tau = min(local_minima, key=lambda tau: cmnd[tau])
        
        # Parabolic interpolation for more precise pitch estimation
        if global_min_tau > 0 and global_min_tau < len(cmnd) - 1:
            x0, x1, x2 = global_min_tau - 1, global_min_tau, global_min_tau + 1
            y0, y1, y2 = cmnd[x0], cmnd[x1], cmnd[x2]
            
            # Compute interpolated minimum
            denom = (y0 - y2)
            if denom != 0:
                interp_tau = x1 + 0.5 * (x0 - x2) * (y0 - y2) / denom
            else:
                interp_tau = global_min_tau
        else:
            interp_tau = global_min_tau
        
        # Convert period to frequency
        f0_value = sr / interp_tau if interp_tau > 0 else np.nan
        
        # Compute voicing probability
        if cmnd[global_min_tau] < threshold:
            # Likely voiced
            voiced_prob = np.exp(-beta * cmnd[global_min_tau])
            f0[frame_idx] = f0_value
            voiced_probabilities[frame_idx] = voiced_prob
        else:
            # Likely unvoiced
            f0[frame_idx] = np.nan
            voiced_probabilities[frame_idx] = 0
    
    # Create voiced flag
    voiced_flag = ~np.isnan(f0)
    
    return f0, voiced_flag, voiced_probabilities
