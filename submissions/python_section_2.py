# Question 9

import pandas as pd
import numpy as np

def calculate_distance_matrix(df)->pd.DataFrame():
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)

    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    np.fill_diagonal(distance_matrix.values, 0)

    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix

# Load the dataset
file_path = r"C:\Users\viyas\OneDrive\Desktop\Projects\Directory\templates\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
df = pd.read_csv(file_path)

distance_matrix = calculate_distance_matrix(df)
    
print(distance_matrix)


# Question 10

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unrolled_data = []

    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


# Load the dataset
file_path = r"C:\Users\viyas\OneDrive\Desktop\Projects\Directory\templates\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
distance_matrix = pd.DataFrame({
        'id_start': ['A', 'A', 'B', 'B', 'C', 'C'],
        'id_end': ['B', 'C', 'A', 'C', 'A', 'B'],
        'distance': [5, 10, 5, 15, 10, 15]
    }).set_index('id_start')


unrolled_df = unroll_distance_matrix(distance_matrix)

print(unrolled_df)


# Question 11


import pandas as pd
import numpy as np

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: str) -> pd.DataFrame:
    if reference_id not in df['id_start'].values:
        raise ValueError(f"Reference ID '{reference_id}' not found in the DataFrame.")

    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    lower_bound = reference_avg_distance * 0.90
    upper_bound = reference_avg_distance * 1.10

    avg_distances = df.groupby('id_start')['distance'].mean()

    within_threshold = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)]

    return within_threshold.reset_index().sort_values(by='distance')


# Load the dataset
file_path = r"C:\Users\viyas\OneDrive\Desktop\Projects\Directory\templates\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
df = pd.read_csv(file_path)
df['id_start'] = df['id_start'].astype(str).str.strip()
df['id_end'] = df['id_end'].astype(str).str.strip()

print("DataFrame Head:")
print(df.head())
print("Unique IDs in id_start:", df['id_start'].unique())

reference_id = 'A'  
try:
    result_df = find_ids_within_ten_percentage_threshold(df, reference_id)
    print("IDs within 10% of the average distance of reference ID:")
    print(result_df)
except ValueError as e:
    print(e)



# Question 12


import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df

# Load the dataset
file_path = r"C:\Users\viyas\OneDrive\Desktop\Projects\Directory\templates\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
df = pd.read_csv(file_path)
df['id_start'] = df['id_start'].astype(str).str.strip()
df['id_end'] = df['id_end'].astype(str).str.strip()

result_df = calculate_toll_rate(df)
print(result_df.head())



# Question 13


import pandas as pd
import datetime

def unroll_distance_matrix(df):
    return df[df['id_start'] != df['id_end']].copy()

def find_ids_within_ten_percentage_threshold(df, reference_id):
    if reference_id not in df['id_start'].values:
        raise ValueError(f"Reference ID '{reference_id}' not found.")
    avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    return df[(df['distance'].between(avg_distance * 0.9, avg_distance * 1.1))][['id_start']].drop_duplicates()

def calculate_toll_rate(df):
    rates = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    return df

def calculate_time_based_toll_rates(df):
    df = df.copy()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Prepare columns for start and end times
    df['start_day'] = ""
    df['start_time'] = datetime.time(0, 0)
    df['end_day'] = ""
    df['end_time'] = datetime.time(23, 59, 59)
    
    for index, row in df.iterrows():
        # Determine the day of the week for the current row
        start_day = days[index % 7]
        df.at[index, 'start_day'] = start_day
        df.at[index, 'end_day'] = start_day

        # Calculate discount based on the day and time ranges
        if start_day in ["Saturday", "Sunday"]:
            discount = 0.7  # Constant for weekends
        else:
            # Weekday discounts
            if index % 24 < 10:  # From 00:00:00 to 10:00:00
                discount = 0.8
            elif index % 24 < 18:  # From 10:00:00 to 18:00
                discount = 1.2
            else:  # From 18:00 to 23:59:59
                discount = 0.8
        
        # Apply the discount to all vehicle types
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            df.at[index, vehicle] *= discount
    
    return df

# Load data from the CSV file
file_path = r"C:\Users\viyas\OneDrive\Desktop\Projects\Directory\templates\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
df = pd.read_csv(file_path)

unrolled_df = unroll_distance_matrix(df)
print("Unrolled Distance Matrix:")
print(unrolled_df)

unique_ids = unrolled_df['id_start'].unique()
print("\nAvailable IDs in 'id_start':", unique_ids)

reference_id = unique_ids[0]  
try:
    ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
    print("\nIDs within 10% threshold of reference ID:")
    print(ids_within_threshold)
except ValueError as e:
    print(e)

toll_df = calculate_toll_rate(unrolled_df)
print("\nToll Rates:")
print(toll_df[['id_start', 'id_end', 'moto', 'car', 'rv', 'bus', 'truck']])

time_based_toll_df = calculate_time_based_toll_rates(toll_df)
time_based_toll_df['distance'] = toll_df['distance']  # Retain the 'distance' column

# Print the desired columns including 'distance'
print("\nTime-Based Toll Rates:")
print(time_based_toll_df[['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']])
