import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

# Function to fix JSON line formatting
def fix_json_line(line):
    line = re.sub(r'([a-zA-Z_]+):', r'"\1":', line)  # Add quotes around the keys
    line = line.rstrip(',')  # Remove any trailing commas
    return line

# Function to load and fix the data
def load_and_fix_data(filepath):
    with open(filepath, 'r') as file:
        data_lines = file.read().strip().split('\n')
    fixed_lines = [fix_json_line(line) for line in data_lines if line.strip() and line.strip() not in ['[', ']']]
    corrected_json_str = f"[{','.join(fixed_lines)}]"
    return json.loads(corrected_json_str)

# Function to plot the data with semi-transparent colors
def plot_data(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='ms')  # Convert from milliseconds
    
    plt.figure(figsize=(15, 7))
    
    # Plot recency, relevance, and finalWeight with semi-transparent colors
    sns.lineplot(data=df, x='timestamp', y='recency', label='Recency', alpha=0.7)
    sns.lineplot(data=df, x='timestamp', y='relevance', label='Relevance', alpha=0.7)
    sns.lineplot(data=df, x='timestamp', y='finalWeight', label='Final Weight', alpha=0.7)

    # Plot nostalgia where it's not null, with semi-transparent color
    df['nostalgia'] = df['nostalgia'].fillna(0)  # Replace nulls with 0 for plotting
    sns.lineplot(data=df, x='timestamp', y='nostalgia', label='Nostalgia', color='purple', linestyle='--', alpha=0.7)
    
    plt.title('Recency, Relevance, Final Weight, and Nostalgia Over Time')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# The path to your JSON file
file_path = 'file.json'  # Replace with your file path

# Load and fix the data
data = load_and_fix_data(file_path)

# Plot the data with semi-transparent colors
plot_data(data)
