import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import numpy as np
import os

def load_dataset(path):
    # Load the dataset
    df = pd.read_csv(path)
    return df

def process_dataset(df):
    """
    Fill NaN values in each column with the mean of that column
    """
    
    # set save path
    save_path = os.path.join('.', 'config', 'column_mean.json')

    # try to load mean of each column from json file
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            column_mean = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        column_mean = None

    # fill NaN values with the mean of that column
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.difference(['id', 'y', 'trickortreat', 'kingofhalloween'])
    for column in numeric_columns:
        if column_mean is not None and column in column_mean:
            df[column] = df[column].fillna(column_mean[column])
        else:
            df[column] = df[column].fillna(df[column].mean())

    # if the mean of each column is not saved, calculate the mean of numeric columns and save it to the json file
    if column_mean is None:
        numeric_means = df[numeric_columns].mean().to_dict()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(numeric_means, f)

def apply_min_max_normalization(df):
    """
    Apply Min-Max normalization to all numeric columns
    """
    chosen_columns = df.select_dtypes(include=['float64', 'int64']).columns.difference(['id', 'y', 'trickortreat', 'kingofhalloween'])

    # Load scaler parameters from JSON file
    try:
        with open('min_max_scaler.json', 'r', encoding='utf-8') as f:
            scaler_params = json.load(f)
            scaler = MinMaxScaler()
            scaler.min_, scaler.scale_ = np.array(scaler_params["min_"]), np.array(scaler_params["scale_"])
            scaler.data_min_, scaler.data_max_, scaler.data_range_ = (
                np.array(scaler_params["data_min_"]),
                np.array(scaler_params["data_max_"]),
                np.array(scaler_params["data_range_"])
            )
    except (FileNotFoundError, json.JSONDecodeError):
        scaler = MinMaxScaler()
        # normalize all chosen columns
        df[chosen_columns] = scaler.fit_transform(df[chosen_columns])
        
        # Save scaler parameters
        scaler_params = {
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
            "data_range_": scaler.data_range_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "min_": scaler.min_.tolist()
        }
        with open(os.path.join('config', 'min_max_scaler.json'), 'w', encoding='utf-8') as f:
            json.dump(scaler_params, f)
    else:
        # Apply loaded scaler parameters
        df[chosen_columns] = scaler.transform(df[chosen_columns])

    return df



def process_words_columns(df, word_dict_path):
    # create a new DataFrame copy to avoid fragmentation
    df = df.copy()
    
    # create all new columns at once
    split_trick = df['trickortreat'].str.split('_', expand=True)
    split_king = df['kingofhalloween'].str.split('_', expand=True)
    
    # load word to idx from json file
    try:
        with open(word_dict_path, 'r') as f:
            word_to_idx = json.load(f)
    except FileNotFoundError:
        print(f"File {word_dict_path} not found")
        exit(1)

    # create new columns
    new_columns = {
        'trick_part1': split_trick[0].map(word_to_idx),
        'trick_part2': split_trick[1].map(word_to_idx),
        'king_part1': split_king[0].map(word_to_idx),
        'king_part2': split_king[1].map(word_to_idx)
    }
    
    # insert new columns after 2rd column
    df.insert(2, 'trick_part1', new_columns['trick_part1'])
    df.insert(3, 'trick_part2', new_columns['trick_part2'])
    df.insert(4, 'king_part1', new_columns['king_part1'])
    df.insert(5, 'king_part2', new_columns['king_part2'])
    
    # delete original columns
    df.drop(columns=['trickortreat', 'kingofhalloween'], inplace=True)
    
    # fill nan values with 0
    for col in ['trick_part1', 'trick_part2', 'king_part1', 'king_part2']:
        df[col] = df[col].fillna(0)
    
    return df

def save_dataset(df, path):
    df.to_csv(path, index=False)


def __main__():
    process_file_name = 'data_sample_xs'
    df = load_dataset(os.path.join('dataset', f'{process_file_name}.csv'))
    process_dataset(df)
    apply_min_max_normalization(df)
    process_words_columns(df, os.path.join('config', 'word_dict.json'))
    save_dataset(df, os.path.join('dataset', f'processed_{process_file_name}.csv'))

if __name__ == '__main__':
    __main__()
