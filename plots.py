import numpy as np
import matplotlib.pyplot as plt
from models import f_T

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