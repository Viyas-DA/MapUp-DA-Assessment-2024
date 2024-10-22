# Question 1

from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []  
    i = 0
    
    while i < len(lst):
        group = lst[i:i+n] 
        
        
        for j in range(len(group) // 2):
            group[j], group[len(group) - 1 - j] = group[len(group) - 1 - j], group[j]
        
        result.extend(group)  
        i += n
        
    return result

print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3)) 
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))            
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4)) 



#Question 2

from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}  
    
    for string in lst:
        length = len(string)  
        
        if length not in length_dict:
            length_dict[length] = [] 
        
       
        length_dict[length].append(string)
    
    
    return dict(sorted(length_dict.items()))
    
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))



#Question 3

from typing import Dict, Any

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here

    flat_dict = {}  
    
    def _flatten(current_key: str, value: Any):
        
        if isinstance(value, dict):
           
            for k, v in value.items():
                new_key = f"{current_key}{sep}{k}" if current_key else k  
                _flatten(new_key, v)  
        elif isinstance(value, list):
          
            for index, item in enumerate(value):
                new_key = f"{current_key}[{index}]"  
                _flatten(new_key, item)  
        else:
            
            flat_dict[current_key] = value
    
    
    _flatten('', nested_dict)
    
    return flat_dict

# Input:
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

print(flatten_dict(nested_dict))

    
# Question 4


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start, end):
        if start == end:
            
            result.add(tuple(nums))  
        else:
            for i in range(start, end):
               
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1, end)
                nums[start], nums[i] = nums[i], nums[start]
    
    result = set()
    backtrack(0, len(nums))
    return [list(perm) for perm in result] 

# Input
nums = [1, 1, 2]
print(unique_permutations(nums))
   


# Question 5

import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """

    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    
   
    matches = re.findall(date_pattern, text)
    
    return matches

# Input
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)



# Question 6

import pandas as pd
import polyline #pip install polyline
from math import radians, sin, cos, sqrt, atan2
from typing import List

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    return distance

    
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    distances = [0]

    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        dist = haversine(lat1, lon1, lat2, lon2)
        distances.append(dist)

    df['distance'] = distances
    
    return df

    
# Example Input

polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
df = polyline_to_dataframe(polyline_str)
print(df)



# Question 7

from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in the same row and column, excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j] 
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j] 
            final_matrix[i][j] = row_sum + col_sum  
    
    return final_matrix

# Input:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)




# Question 8

import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    
    day_to_date = {
        'Monday': '2024-10-14',   
        'Tuesday': '2024-10-15',
        'Wednesday': '2024-10-16',
        'Thursday': '2024-10-17',
        'Friday': '2024-10-18',
        'Saturday': '2024-10-19',
        'Sunday': '2024-10-20'
    }

    # Replace the day strings with actual dates
    df['startDay'] = df['startDay'].map(day_to_date)
    df['endDay'] = df['endDay'].map(day_to_date)

    # Convert to datetime
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    grouped = df.groupby(['id', 'id_2'])

    results = pd.Series(index=grouped.groups.keys(), dtype=bool)

    for (id_value, id_2_value), group in grouped:
        start_times = group['start']
        end_times = group['end']
        
        
        full_week_covered = (start_times.dt.dayofweek.min() == 0 and start_times.dt.dayofweek.max() == 6)
        
        full_day_covered = (start_times.min().time() == pd.Timestamp('00:00:00').time() and
                            end_times.max().time() == pd.Timestamp('23:59:59').time())

        
        results[(id_value, id_2_value)] = not (full_week_covered and full_day_covered)

    return results


# Sample DataFrame
data = {
        'id': [1040000, 1040010, 1040020, 1040030],
        'id_2': [-1, -1, -1, -1],
        'startDay': ['Monday', 'Monday', 'Thursday', 'Monday'],
        'startTime': ['05:00:00', '10:00:00', '15:00:00', '19:00:00'],
        'endDay': ['Wednesday', 'Friday', 'Friday', 'Friday'],
        'endTime': ['10:00:00', '15:00:00', '19:00:00', '23:59:59'],
    }
    
df = pd.DataFrame(data)
result = time_check(df)
print(result)
