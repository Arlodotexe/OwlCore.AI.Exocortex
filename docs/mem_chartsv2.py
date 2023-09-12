import numpy as np
import matplotlib.pyplot as plt

# Constants
LONG_TERM_DECAY_THRESHOLD = 0.1

# Decay functions
def short_term_decay(t, short_term_decay_rate):
    """Compute the short-term decay."""
    return np.exp(-short_term_decay_rate * t)

def long_term_decay(t, a, b, c):
    """Compute the adjusted long-term decay using a reversed logarithmic function."""
    return a - b * np.log(c * t + 1)

def compute_short_term_decay_threshold(long_term_duration):
    """Compute the short-term decay threshold based on an exponential function."""
    k = 0.00001
    return 0.2 + 0.8 * np.exp(-k * long_term_duration)

# Parameters for the plots
short_term_duration = 25  # in minutes
short_term_duration_hours = short_term_duration / 60  # Convert to hours for plotting purposes

long_term_durations_in_hours = {
    "Infancy to Early Childhood": 5 * 365 * 24,
    "Childhood to Adolescence": (18 - 5) * 365 * 24,
    "Young Adulthood": (30 - 18) * 365 * 24,
    "Middle Age": 40 * 365 * 24,
    "Elderly": 70 * 365 * 24
}

def memory_strength_over_time(t, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength):
    """Compute the memory strength over time, transitioning from short-term to long-term decay."""
    short_term_strength = short_term_decay(t, short_term_decay_rate)
    adjusted_t = t - short_term_duration_hours
    offset = transition_strength - long_term_decay(0, a, b, c)
    long_term_strength = long_term_decay(adjusted_t, a, b, c) + offset
    combined_strength = np.where(t <= short_term_duration_hours, short_term_strength, long_term_strength)
    return np.maximum(LONG_TERM_DECAY_THRESHOLD, combined_strength)

def generate_combined_plot_for_stage(title, long_term_duration, short_term_duration):
    """Generate a combined 2x2 plot for the given life stage."""
    short_term_decay_threshold = compute_short_term_decay_threshold(long_term_duration)
    short_term_decay_rate = -np.log(short_term_decay_threshold) / short_term_duration_hours
    a = short_term_decay_threshold
    b = (a - LONG_TERM_DECAY_THRESHOLD) / np.log(long_term_duration)
    c = 1
    transition_strength = short_term_decay(short_term_duration_hours, short_term_decay_rate)

    t_short_1_5x = np.linspace(0, 1.5 * short_term_duration_hours, 1000)
    t_short_5x = np.linspace(0, 5 * short_term_duration_hours, 1000)
    t_first_0_33x = np.linspace(0, 0.33 * long_term_duration, 1000)
    t_full_span = np.linspace(0, long_term_duration, 1000)

    memory_strength_short_1_5x = memory_strength_over_time(t_short_1_5x, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_short_5x = memory_strength_over_time(t_short_5x, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_first_0_33x = memory_strength_over_time(t_first_0_33x, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_full_span = memory_strength_over_time(t_full_span, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Memory Decay Over Time - {title}", fontsize=16)
    
    time_arrays = [t_short_1_5x, t_short_5x, t_first_0_33x, t_full_span]
    memory_strengths = [memory_strength_short_1_5x, memory_strength_short_5x, memory_strength_first_0_33x, memory_strength_full_span]
    titles = ["1.5x Short-Term Duration", "5x Short-Term Duration", "First 0.33x Total Duration", "Full Duration"]
    x_labels = ["Time (hours)", "Time (hours)", "Time (years)", "Time (years)"]

    for i, ax in enumerate(axes.ravel()):
        ax.plot(time_arrays[i], memory_strengths[i], label="Recency Weight", color="blue")
        
        # Only start the "Nostalgia Curve" where it intersects with the "Recency Weight"
        filtered_time_values = time_arrays[i][memory_strengths[i] <= 1 - memory_strengths[i]]
        
        # If filtered_time_values is not empty, compute the nostalgia_curve_start
        if len(filtered_time_values) > 0:
            nostalgia_curve_start = np.min(filtered_time_values)
            t_nostalgia = time_arrays[i][time_arrays[i] >= nostalgia_curve_start]
            ax.plot(t_nostalgia, 1 - memory_strength_over_time(t_nostalgia, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength), 
                    label="Nostalgia Curve", color="orange")
            
        ax.axvline(x=short_term_duration_hours, color="red", linestyle="--", label="End of Short-Term Duration")
        ax.axhline(y=short_term_decay_threshold, color="green", linestyle="--", label="Short-Term Decay Threshold")
        ax.axhline(y=LONG_TERM_DECAY_THRESHOLD, color="purple", linestyle="--", label="Long-Term Decay Threshold")
        ax.set_ylim([0, 1])
        
        if "years" in x_labels[i]:
            x_ticks_hours = np.linspace(0, max(time_arrays[i]), 6)
            x_ticks_years = x_ticks_hours / (365 * 24)
            ax.set_xticks(x_ticks_hours)
            ax.set_xticklabels(np.round(x_ticks_years, 2))
        
        ax.set_title(titles[i])
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel("Weight")
        ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    combined_file_path = f"./combined_memory_decay_2x2_{title.replace(' ', '_')}.png"
    plt.savefig(combined_file_path)
    plt.close(fig)
    
    return combined_file_path

# Loop to generate combined plot for all stages
combined_plot_paths_for_all_stages = []

for stage_name, duration in long_term_durations_in_hours.items():
    plot_path = generate_combined_plot_for_stage(stage_name, duration, short_term_duration)
    combined_plot_paths_for_all_stages.append((stage_name, plot_path))

combined_plot_paths_for_all_stages
