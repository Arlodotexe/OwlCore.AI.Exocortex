import numpy as np
import matplotlib.pyplot as plt

# Constants
LONG_TERM_DECAY_THRESHOLD = 0.01

# Decay functions
# The short-term memory decay is modeled using an exponential function, which captures the rapid fading of recent memories.
# This is consistent with how short-term human memories fade quickly.
def short_term_decay(t, short_term_decay_rate):
    """Compute the short-term decay."""
    return np.exp(-short_term_decay_rate * t)

# The long-term memory decay is modeled using a reversed logarithmic function.
# This is based on the observation that human memories fade slowly over time but never completely disappear.
# The initial rapid decay slows down, representing the long-lasting nature of certain memories.
def long_term_decay(t, a, b, c):
    """Compute the adjusted long-term decay using a reversed logarithmic function."""
    return a - b * np.log(c * t + 1)

# The short-term decay threshold is computed using a logarithmic function.
# This threshold represents the point at which short-term memories start transitioning to long-term memories.
# The function ensures that as more long-term memories accumulate, the threshold decreases, 
# leading short-term memories to decay faster.
def compute_short_term_decay_threshold(long_term_duration):
    """Compute the short-term decay threshold based on an exponential function."""
    k = 0.00001
    return 0.2 + 0.8 * np.exp(-k * long_term_duration)

# Parameters for the plots
short_term_duration = 5  # in minutes
short_term_duration_hours = short_term_duration / 60  # Convert to hours for plotting purposes

# The long-term durations in hours for different life stages serve as benchmarks to evaluate the memory model.
long_term_durations_in_hours = {
    "Infancy to Early Childhood": 5 * 365 * 24,
    "Childhood to Adolescence": (18 - 5) * 365 * 24,
    "Young Adulthood": (30 - 18) * 365 * 24,
    "Middle Age": 40 * 365 * 24,
    "Elderly": 70 * 365 * 24
}

# List to store file paths for generated plots
adjusted_plot_paths = []

# Create plots for each long-term duration
for title, long_term_duration in long_term_durations_in_hours.items():
    
    # Create figure with 2x3 layout
    fig, axes = plt.subplots(3, 2, figsize=(21, 12))
    fig.suptitle(f"Memory Decay Over Time - {title}", fontsize=16)

    # Calculate short-term decay threshold
    short_term_decay_threshold = compute_short_term_decay_threshold(long_term_duration)
    short_term_decay_rate = -np.log(short_term_decay_threshold) / short_term_duration_hours

    # Parameters for the reversed logarithmic long-term decay
    a = short_term_decay_threshold
    b = 1
    c = 1 / (long_term_duration / np.log(1 / LONG_TERM_DECAY_THRESHOLD - 1))
    
    # Parameters for the reversed logarithmic long-term decay
    a = short_term_decay_threshold  # Start from where short-term decay ends
    b = (a - LONG_TERM_DECAY_THRESHOLD) / np.log(long_term_duration)  # Adjust to approach the threshold at max duration
    c = 1  # A default value, can be adjusted if needed
    
    # Get the value at the end of the short term duration to use as a factor for long term decay.
    transition_strength = short_term_decay(short_term_duration_hours, short_term_decay_rate)
    
    # Time arrays for plots
    t_short_1_5x = np.linspace(0, 1.5 * short_term_duration_hours, 1000)
    t_short_5x = np.linspace(0, 5 * short_term_duration_hours, 1000)
    t_long_0_5x = np.linspace(long_term_duration - 0.5 * long_term_duration, long_term_duration, 1000)  # Last 0.5x of long-term duration
    t_long_10th_year = np.linspace(10 * 365 * 24 - 365 * 24, 10 * 365 * 24, 1000)  # 10th year
    t_full_span = np.linspace(0, long_term_duration, 1000)
    
    # Modify the decay functions to handle the transition between short-term and reversed-log long-term decay
    def memory_strength_over_time(t, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength):
        """Compute the memory strength over time, transitioning from short-term to long-term decay."""
        # Short-term decay for times <= short-term duration
        short_term_strength = short_term_decay(t, short_term_decay_rate)
        
        # Long-term decay for times > short-term duration
        adjusted_t = t - short_term_duration_hours
        offset = transition_strength - long_term_decay(0, a, b, c)
        long_term_strength = long_term_decay(adjusted_t, a, b, c) + offset
        
        # Combine the two using a piecewise function
        combined_strength = np.where(t <= short_term_duration_hours, short_term_strength, long_term_strength)
        
        return np.maximum(LONG_TERM_DECAY_THRESHOLD, combined_strength)

    # Modify the memory strength arrays computation
    memory_strength_short_1_5x = memory_strength_over_time(t_short_1_5x, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_short_5x = memory_strength_over_time(t_short_5x, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_long_0_5x = memory_strength_over_time(t_long_0_5x, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_long_10th_year = memory_strength_over_time(t_long_10th_year, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)
    memory_strength_full_span = memory_strength_over_time(t_full_span, short_term_decay_rate, a, b, c, short_term_duration_hours, transition_strength)

    # Plotting
    # 1.5x short term duration
    axes[0][0].plot(t_short_1_5x, memory_strength_short_1_5x, label="Memory Strength", color="blue")
    axes[0][0].axvline(x=short_term_duration_hours, color="red", linestyle="--", label="End of Short-Term Duration")
    axes[0][0].axhline(y=short_term_decay_threshold, color="green", linestyle="--", label="Short-Term Decay Threshold")
    axes[0][0].axhline(y=LONG_TERM_DECAY_THRESHOLD, color="purple", linestyle="--", label="Long-Term Decay Threshold")
    axes[0][0].set_ylim([0, 1])
    axes[0][0].set_xlim([0, 1.5 * short_term_duration_hours])
    axes[0][0].set_title(f"Memory Decay over 1.5x Short-Term Duration")
    axes[0][0].set_xlabel("Time (hours)")
    axes[0][0].set_ylabel("Memory Strength")
    axes[0][0].legend()

    # 5x short term duration
    axes[0][1].plot(t_short_5x, memory_strength_short_5x, label="Memory Strength", color="blue")
    axes[0][1].axvline(x=short_term_duration_hours, color="red", linestyle="--", label="End of Short-Term Duration")
    axes[0][1].axhline(y=short_term_decay_threshold, color="green", linestyle="--", label="Short-Term Decay Threshold")
    axes[0][1].axhline(y=LONG_TERM_DECAY_THRESHOLD, color="purple", linestyle="--", label="Long-Term Decay Threshold")
    axes[0][1].set_ylim([0, 1])
    axes[0][1].set_xlim([0, 5 * short_term_duration_hours])
    axes[0][1].set_title(f"Memory Decay over 5x Short-Term Duration")
    axes[0][1].set_xlabel("Time (hours)")
    axes[0][1].set_ylabel("Memory Strength")
    axes[0][1].legend()


    # 0.5x long term duration
    axes[1][0].plot(t_long_0_5x, memory_strength_long_0_5x, label="Memory Strength", color="blue")
    axes[1][0].axhline(y=LONG_TERM_DECAY_THRESHOLD, color="purple", linestyle="--", label="Long-Term Decay Threshold")
    axes[1][0].axvline(x=long_term_duration, color="red", linestyle="--", label="End of Long-Term Duration")
    axes[1][0].set_ylim([0, 1])
    axes[1][0].set_xlim([long_term_duration - 0.5 * long_term_duration, long_term_duration])
    axes[1][0].set_xticks(np.linspace(long_term_duration - 0.5 * long_term_duration, long_term_duration, 6))
    axes[1][0].set_title("Memory Decay over Last 0.5x Long-Term Duration")
    axes[1][0].set_xlabel("Time (years)")
    axes[1][0].set_ylabel("Memory Strength")
    axes[1][0].legend()

    # Convert x-ticks from hours to years for better readability
    x_ticks_hours = axes[1][0].get_xticks()
    x_ticks_years = x_ticks_hours / (365 * 24)
    axes[1][0].set_xticks(x_ticks_hours)
    axes[1][0].set_xticklabels(np.round(x_ticks_years, 2))

    # 10th year
    axes[1][1].plot(t_long_10th_year, memory_strength_long_10th_year, label="Memory Strength", color="blue")
    axes[1][1].axhline(y=short_term_decay_threshold, color="green", linestyle="--", label="Short-Term Decay Threshold")
    axes[1][1].axhline(y=LONG_TERM_DECAY_THRESHOLD, color="purple", linestyle="--", label="Long-Term Decay Threshold")
    axes[1][1].axvline(x=long_term_duration, color="red", linestyle="--", label="End of Long-Term Duration")
    axes[1][1].set_ylim([0, 1])
    axes[1][1].set_xlim([10 * 365 * 24 - 365 * 24, 10 * 365 * 24])
    axes[1][1].set_xticks(np.linspace(10 * 365 * 24 - 365 * 24, 10 * 365 * 24, 13))
    axes[1][1].set_xticklabels([f"{(i * 2) - i}" for i in range(0, 13)])
    axes[1][1].set_title("Memory Decay over Last Year")
    axes[1][1].set_xlabel("Time (months)")
    axes[1][1].set_ylabel("Memory Strength")
    axes[1][1].legend()

    # Zoomed-out plot covering the full span of the long-term duration
    axes[2][0].plot(t_full_span, memory_strength_full_span, label="Memory Strength", color="blue")
    axes[2][0].axhline(y=short_term_decay_threshold, color="green", linestyle="--", label="Short-Term Decay Threshold")
    axes[2][0].axhline(y=LONG_TERM_DECAY_THRESHOLD, color="purple", linestyle="--", label="Long-Term Decay Threshold")
    axes[2][0].axvline(x=short_term_duration_hours, color="red", linestyle="--", label="End of Short-Term Duration")
    axes[2][0].axvline(x=long_term_duration, color="red", linestyle="--", label="End of Long-Term Duration")
    axes[2][0].set_ylim([0, 1])
    axes[2][0].set_xlim([0, long_term_duration])
    axes[2][0].set_title("Memory Decay over Full Span")
    axes[2][0].set_xlabel("Time (years)")
    axes[2][0].set_ylabel("Memory Strength")
    axes[2][0].legend()

    # Convert x-ticks from hours to years for better readability
    x_ticks_hours_full = np.linspace(0, long_term_duration, 11)
    x_ticks_years_full = x_ticks_hours_full / (365 * 24)
    axes[2][0].set_xticks(x_ticks_hours_full)
    axes[2][0].set_xticklabels(np.round(x_ticks_years_full, 2))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save the plot to a file and store the path
    adjusted_file_path = f"./adjusted_memory_decay_{title.replace(' ', '_')}.png"
    plt.savefig(adjusted_file_path)
    adjusted_plot_paths.append(adjusted_file_path)

    # Close the figure to free up memory
    plt.close(fig)

adjusted_plot_paths