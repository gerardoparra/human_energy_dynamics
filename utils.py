import matplotlib.pyplot as plt
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



## Plot related
def set_day_xticks(t_span, ax=None):
    hours = np.arange(t_span[0],t_span[1]+1,24)
    days  = (hours/24).astype(int)
    if ax is not None:
        ax.set_xticks(hours, days)
    else:
        plt.xticks(hours, days)

def plot_days(days, ax=None, amp=(-1.1,1)):
    if ax is not None:
        ax.vlines(days, amp[0], amp[1], colors='#ccc', linestyles='dashed')
    else:
        plt.vlines(days, amp[0], amp[1], colors='#ccc', linestyles='dashed')

# Make plots of circadian rhythm solutions
def plot_circadian(solution, ax=None, figsize=(6,3)):
    # Old: rhythm = (1 -np.cos(solution.y[0])) / 2 
    rhythm = (1 - np.cos(2*np.pi/solution.y[0] * solution.t)) / 2

    new_fig = ax is None  # Check if we need to create a new figure
    if new_fig:
        fig, ax = plt.subplots(figsize=figsize)  # Create new figure if no axis is given
    else:
        fig = ax.get_figure()  # Get the parent figure of the existing axis

    ax.plot(solution.t, rhythm, 'k')
    ax.set_ylabel("C(t)")

    if new_fig:
        plt.show()  # Show only if a new figure was created

    return fig, ax  # Return figure and axis for customization