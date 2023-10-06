import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
# These are real-world values extracted from a research prototype 3 weeks old.
LONG_TERM_DECAY_THRESHOLD = 0.01
TOTAL_MEMORIES_3WEEKS = 1622
CORE_MEMORIES_3WEEKS = 207
RECOLLECTIONS_3WEEKS = 1210
REACTIONS_3WEEKS = 205

THREE_WEEKS_IN_HOURS = 3 * 7 * 24
NUM_RUNS = 5  # Number of runs to average over

# Adjust this fraction as needed to control the nostalgia boost
nostalgia_boost_fraction = 0.999

# Age ranges in hours
age_ranges = {
    "6 hours": 6,
    "1 day": 24,
    "7 days": 7 * 24,
    "1 month": 30 * 24,
    "6 months": 6 * 30 * 24,
    "1 year": 365 * 24,
    "2 years": 2 * 365 * 24, 
    "5 years": 5 * 365 * 24,
    "10 years": 10 * 365 * 24,
    "15 years": 15 * 365 * 24,
    "25 years": 25 * 365 * 24,
    "50 years": 50 * 365 * 24,
    "75 years": 75 * 365 * 24,
    "100 years": 100 * 365 * 24,
    "250 years": 250 * 365 * 24,
    "500 years": 500 * 365 * 24,
    "1000 years": 1000 * 365 * 24,
}

def compute_short_term_decay_threshold(long_term_duration):
    """Compute the short-term decay threshold based on the provided formula."""
    T_max = 1 - LONG_TERM_DECAY_THRESHOLD
    T_min = LONG_TERM_DECAY_THRESHOLD
    D_lt = long_term_duration / 24  # Convert hours to days for consistency with the C# code
    return T_min + T_max * np.exp(-0.00001 * D_lt)

def memory_strength(t, short_term_decay_rate, short_term_duration_hours):
    """Compute the memory strength based on recency."""
    short_term_strength = np.exp(-short_term_decay_rate * t)
    return np.where(t <= short_term_duration_hours, short_term_strength, LONG_TERM_DECAY_THRESHOLD)

def retrieve_memories(short_term_decay_threshold, working_recollection_memory_distance_threshold):
    """Simulate memory retrieval based on thresholds."""
    recency_weights = memory_strength(memories_age, short_term_decay_rate, short_term_duration_hours)
    nostalgia_curve = 1 - recency_weights
    
    # Calculate the WorkingRecollectionMemoryWeightThreshold
    working_recollection_memory_weight_threshold = 1 - ((1 - short_term_decay_threshold) * working_recollection_memory_distance_threshold)
    
    combined_weights = recency_weights + (nostalgia_curve * memories_relevancy) * working_recollection_memory_distance_threshold
    
    return np.sum(combined_weights > working_recollection_memory_weight_threshold)

results = {}

plt.figure(figsize=(15, 10))
colors = plt.cm.jet(np.linspace(0, 1, len(age_ranges)))

for idx, (age_name, max_age) in enumerate(age_ranges.items()):
    retrieved_counts = []
    scale_factor = max_age / THREE_WEEKS_IN_HOURS
    TOTAL_MEMORIES = int(TOTAL_MEMORIES_3WEEKS * scale_factor)
    CORE_MEMORIES = int(CORE_MEMORIES_3WEEKS * scale_factor)
    RECOLLECTIONS = int(RECOLLECTIONS_3WEEKS * scale_factor)
    REACTIONS = int(REACTIONS_3WEEKS * scale_factor)
    
    short_term_duration_hours = 8
    short_term_decay_threshold = compute_short_term_decay_threshold(max_age)
    short_term_decay_rate = -np.log(short_term_decay_threshold) / short_term_duration_hours

   # Adjust the threshold to be a fraction of the distance into short-term memory
    working_recollection_memory_distance_threshold = nostalgia_boost_fraction * short_term_decay_threshold

    for run_idx in range(NUM_RUNS):
        np.random.seed(run_idx)  # Set seed for reproducibility
        memories_age = np.random.rand(TOTAL_MEMORIES) * max_age
        memories_relevancy = np.random.rand(TOTAL_MEMORIES)
        retrieved_counts.append(retrieve_memories(short_term_decay_threshold, working_recollection_memory_distance_threshold))

    avg_retrieved = np.mean(retrieved_counts)
    
    if avg_retrieved == 0:
        lower_bound = 0
        upper_bound = short_term_decay_threshold
        while retrieve_memories(short_term_decay_threshold, upper_bound) < 10:
            upper_bound *= 2
    else:
        lower_bound = short_term_decay_threshold
        upper_bound = short_term_decay_threshold

    results[age_name] = {
        "ShortTermDecayThreshold": short_term_decay_threshold,
        "AvgMemoriesRetrievedWithDefault": avg_retrieved,
        "LowerBound": lower_bound,
        "UpperBound": upper_bound
    }

# Plotting the results
ages = list(age_ranges.keys())
retrieved_counts = [results[age]['AvgMemoriesRetrievedWithDefault'] for age in ages]
lower_bounds = [results[age]['LowerBound'] for age in ages]
upper_bounds = [results[age]['UpperBound'] for age in ages]

# Outputting results in a table
print("Age Range".ljust(20), "ShortTermDecayThreshold".ljust(30), "AvgMemoriesRetrievedWithDefault".ljust(35), "LowerBound".ljust(20), "UpperBound".ljust(20))
print('-'*125)
for age_name in age_ranges.keys():
    print(age_name.ljust(20), 
          str(results[age_name]['ShortTermDecayThreshold']).ljust(30), 
          str(results[age_name]['AvgMemoriesRetrievedWithDefault']).ljust(35), 
          str(results[age_name]['LowerBound']).ljust(20), 
          str(results[age_name]['UpperBound']).ljust(20))

plt.tight_layout()
# plt.show()