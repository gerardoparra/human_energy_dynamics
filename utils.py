import numpy as np

## Model related
# Create input repeating every 24h for integrative models
def input_daily(t, params):
    """
    Determines light input based on time of day with configurable hours and intensity.
    
    Parameters:
    - t: scalar or numpy array of time values (hours).
    - params: Dictionary with:
        - 'hours': List of tuples [(start, end)] defining light-on periods.
        - 'amp': Single amplitude value or list of amplitudes (one per time range).
    
    Returns:
    - A numpy array (or scalar) indicating light presence, scaled by amplitude.
    """
    t_mod = np.mod(t, 24)  # Convert to 24-hour cycle
    light = np.zeros_like(t_mod, dtype=float)

    # Ensure amp is a list, or convert it to a list of the same value
    amps = params['amp']
    if not isinstance(amps, (list, np.ndarray)):  
        amps = [amps] * len(params['hours'])  # Repeat single amp for all peaks
    
    for (start, end), amp in zip(params['hours'], amps):
        light += amp * ((t_mod >= start) & (t_mod < end))

    return light

def input_daily_smooth(t, params):
    """
    Generates a smooth, daily repeating light input (e.g. Gaussian-shaped daylight curve).

    Parameters:
    - t: scalar or numpy array of time values (in hours).
    - params: Dictionary with keys:
        - 'center': Center hour of light input peak (e.g. 12 for noon)
        - 'amp': Maximum amplitude of light (e.g. 1.0)
        - 'width': Spread of the curve (controls duration of "day") in hours

    Returns:
    - Array or scalar of light input values.
    """
    t_mod = np.mod(t, 24)  # Wrap time to 24-hour clock
    center = params.get('center', 12)  # Default to noon
    amp = params.get('amp', 1.0)
    width = params.get('width', 3.0)  # Default spread of ~6 hours (FWHM ~ 2.35*width)

    # Gaussian profile centered at `center` with spread `width`
    light = amp * np.exp(-0.5 * ((t_mod - center) / width) ** 2)

    return light

def input_variable(t, params):
    """
    Determines external input based on a schedule that varies daily.
    
    Parameters:
    - t: scalar or numpy array of time values (hours).
    - params: Dictionary with:
        - 'hours': List of lists, where each inner list contains tuples [(start, end)] defining light-on periods for that day.
        - 'amp': List of lists, where each inner list contains amplitude values corresponding to the time ranges in 'hours'.
    
    Returns:
    - A numpy array (or scalar) indicating input presence, scaled by amplitude.
    """
    days_elapsed = np.floor(t / 24).astype(int)  # Compute which day it is
    t_mod = np.mod(t, 24)  # Convert to 24-hour format
    input_signal = np.zeros_like(t_mod, dtype=float)

    for i, day_hours in enumerate(params['hours']):
        if i >= len(params['hours']):  
            break  # Prevent out-of-bounds errors
        
        if i >= len(params['amp']):  
            break  # Prevent mismatch between hours and amplitudes
        
        amps = params['amp'][i]  # Get amplitudes for the i-th day

        # Make sure the number of amplitudes matches the number of time ranges
        if not isinstance(amps, (list, np.ndarray)):
            amps = [amps] * len(day_hours)  # Repeat single amp for all peaks

        mask = days_elapsed == i  # Select times corresponding to this day

        for (start, end), amp in zip(day_hours, amps):
            input_signal[mask] += amp * ((t_mod[mask] >= start) & (t_mod[mask] < end))

    return input_signal

def get_intervals(time, state):
    """
    Given a 1D array of time and a boolean state array, return a list of intervals
    where state is True.
    """
    intervals = []
    in_interval = False
    start = None
    for t, s in zip(time, state):
        if s and not in_interval:
            in_interval = True
            start = t
        elif not s and in_interval:
            in_interval = False
            intervals.append((start, t))
    if in_interval:
        intervals.append((start, time[-1]))
    return intervals