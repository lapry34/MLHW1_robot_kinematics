import pandas as pd
import numpy as np
import sys
import os

# Set random seed for reproducibility
rng = np.random.default_rng(seed=1979543)

# Directory containing CSV files
input_dir = 'dataset/logs'
output_dir = 'dataset/data'  # Directory to save modified files

# Function to add noise and save modified files
def generate_dataset(noise, noise_percent):
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(input_dir, filename), delimiter=';')

            # If noise is True, add random noise to a subset of rows in numeric columns
            if noise:
                noise_level = 0.01  # Adjust noise level as needed
                noisy_df = df.copy()

                # Determine number of rows to apply noise
                num_noisy_rows = int((noise_percent / 100) * len(noisy_df))
                
                # Select random rows to add noise
                noisy_rows = rng.choice(noisy_df.index, size=num_noisy_rows, replace=False)
                
                # Apply noise to selected rows and numeric columns
                for col in noisy_df.select_dtypes(include=np.number).columns:
                    noisy_df.loc[noisy_rows, col] += rng.normal(0, noise_level, size=num_noisy_rows)

                # Round the noisy data to 3 decimal places
                noisy_df = noisy_df.round(3)

                # Append the noisy rows to the original DataFrame
                combined_df = pd.concat([df, noisy_df.loc[noisy_rows]], ignore_index=True)

                # Extract last two characters of the filename for the new filename
                new_filename = f"dataset_{filename[-6:-4]}.csv"  
                combined_df.to_csv(os.path.join(output_dir, new_filename), index=False, sep=';')

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    generate_dataset(noise=True, noise_percent=20)  # Generate the modified dataset
    sys.exit(0)
