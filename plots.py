import numpy as np
import matplotlib.pyplot as plt
from models import *
from scipy.integrate import solve_ivp

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
    # rhythm = (1 - np.cos(2*np.pi/solution.y[0] * solution.t)) / 2
    rhythm = f_T(solution.y[0], solution.t)

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

# Plot batches of energy simiulations
def plot_energy_conditions(params_energy, t_span, t_eval, y0, days):
    """
    Plots energy dynamics across batches of light and rest conditions.
    Instead of plotting a line for rest, the background is colored white for awake
    and light gray for asleep.
    """
    C_fixed = {k: v for k, v in params_energy['C'].items() if k != 'light_conditions'}
    S_fixed = {k: v for k, v in params_energy['S'].items() if k != 'rest_conditions'}
    light_conditions = params_energy['C']['light_conditions']
    rest_conditions = params_energy['S']['rest_conditions']
    
    num_plots = min(len(light_conditions), len(rest_conditions))
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    # Choose a threshold for "asleep" based on your rest function values.
    sleep_threshold = 0.5
    
    for i in range(num_plots):
        light_cond = light_conditions[i]
        rest_cond = rest_conditions[i]

        params = {
            'C': {**C_fixed, 'light': light_cond['light']},
            'S': {**S_fixed, 'rest': rest_cond['rest']}
        }

        sol = solve_ivp(energy, t_span, y0, t_eval=t_eval, args=(params,), rtol=1e-8, atol=1e-10)

        ax = axes[i]
        plot_days(days, amp=(-1.75, f_S(sol.y[1]).max()), ax=ax)
        ax.hlines([0,1], 0, t_span[-1], '#ccc', ':')

        # Plot sleepiness and circadian rhythm as before
        ax.plot(sol.t, f_S(sol.y[1]), 'k', label='Sleepiness')
        ax.plot(sol.t, f_T(sol.y[0], sol.t)/2 - 0.75, label='Circadian rhythm')
        ax.plot(sol.t, params['C']['light']['L'](sol.t, params['C']['light'])/2 - 1.75, label='Light input', color='orange')

        ax.set_yticks([0, 1])
        
        # Compute rest values and determine sleep intervals.
        R_vals = params['S']['rest']['R'](sol.t, params['S']['rest'])
        # Define sleep state: asleep if R_vals > threshold.
        sleep_state = R_vals > sleep_threshold
        intervals = get_intervals(sol.t, sleep_state)
        # Shade intervals where the subject is asleep (light gray) and leave awake (white).
        for (start, end) in intervals:
            ax.axvspan(start, end, color='lightgray', alpha=0.5, zorder=0)
        
        # Title logic
        c_title = light_cond.get('title', f'C{i+1}')
        s_title = rest_cond.get('title', f'S{i+1}')
        ax.set_title(f'{c_title} - {s_title}')
        if i == num_plots - 1:
            ax.set_xlabel('Time [days]')

    set_day_xticks(t_span, ax=axes[-1])
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()