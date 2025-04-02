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

## Plot related
def set_day_xticks(t_span, ax=None):
    if ax is not None:
        ax.set_xticks(np.arange(t_span[0],t_span[1],24), np.arange(t_span[0]/24,t_span[1]/24,1).astype(int))
    else:
        plt.xticks(np.arange(t_span[0],t_span[1],24), np.arange(t_span[0]/24,t_span[1]/24,1).astype(int))

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