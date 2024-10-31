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
    Apply Min-Max normalization:
    - y column: scale to [0, 1]
    - trick_part1, trick_part2, king_part1, king_part2: scale to [0, 1]
    - other numeric columns: scale to [-1, 1]
    """
    # 选择需要归一化到[-1,1]的特征列
    feature_columns = df.select_dtypes(include=['float64', 'int64']).columns.difference(
        ['id', 'y', 'trick_part1', 'trick_part2', 'king_part1', 'king_part2']
    )
    
    # 需要归一化到[0,1]的列
    word_columns = ['trick_part1', 'trick_part2', 'king_part1', 'king_part2']
    
    try:
        # 加载特征的scaler参数
        with open(os.path.join('config', 'feature_scaler.json'), 'r', encoding='utf-8') as f:
            feature_params = json.load(f)
            feature_scaler = MinMaxScaler(feature_range=(-1, 1))
            feature_scaler.min_, feature_scaler.scale_ = np.array(feature_params["min_"]), np.array(feature_params["scale_"])
            feature_scaler.data_min_, feature_scaler.data_max_, feature_scaler.data_range_ = (
                np.array(feature_params["data_min_"]),
                np.array(feature_params["data_max_"]),
                np.array(feature_params["data_range_"])
            )
            
        # 加载y和word列的scaler参数
        with open(os.path.join('config', 'zero_one_scaler.json'), 'r', encoding='utf-8') as f:
            zero_one_params = json.load(f)
            zero_one_scaler = MinMaxScaler(feature_range=(0, 1))
            zero_one_scaler.min_ = np.array(zero_one_params["min_"])
            zero_one_scaler.scale_ = np.array(zero_one_params["scale_"])
            zero_one_scaler.data_min_ = np.array(zero_one_params["data_min_"])
            zero_one_scaler.data_max_ = np.array(zero_one_params["data_max_"])
            zero_one_scaler.data_range_ = np.array(zero_one_params["data_range_"])
            
    except (FileNotFoundError, json.JSONDecodeError):
        # 特征归一化到[-1, 1]
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        feature_values = df[feature_columns].values
        df[feature_columns] = feature_scaler.fit_transform(feature_values)
        
        # y和word列归一化到[0, 1]
        zero_one_scaler = MinMaxScaler(feature_range=(0, 1))
        zero_one_columns = ['y'] + word_columns
        zero_one_values = df[zero_one_columns].values
        df[zero_one_columns] = zero_one_scaler.fit_transform(zero_one_values)
        
        # 保存特征的scaler参数
        feature_params = {
            "data_min_": feature_scaler.data_min_.tolist(),
            "data_max_": feature_scaler.data_max_.tolist(),
            "data_range_": feature_scaler.data_range_.tolist(),
            "scale_": feature_scaler.scale_.tolist(),
            "min_": feature_scaler.min_.tolist()
        }
        with open(os.path.join('config', 'feature_scaler.json'), 'w', encoding='utf-8') as f:
            json.dump(feature_params, f)
            
        # 保存[0,1]归一化的scaler参数
        zero_one_params = {
            "data_min_": zero_one_scaler.data_min_.tolist(),
            "data_max_": zero_one_scaler.data_max_.tolist(),
            "data_range_": zero_one_scaler.data_range_.tolist(),
            "scale_": zero_one_scaler.scale_.tolist(),
            "min_": zero_one_scaler.min_.tolist()
        }
        with open(os.path.join('config', 'zero_one_scaler.json'), 'w', encoding='utf-8') as f:
            json.dump(zero_one_params, f)
    else:
        # 应用已保存的scaler参数
        feature_values = df[feature_columns].values
        df[feature_columns] = feature_scaler.transform(feature_values)
        
        zero_one_columns = ['y'] + word_columns
        zero_one_values = df[zero_one_columns].values
        df[zero_one_columns] = zero_one_scaler.transform(zero_one_values)

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

    # 定义映射函数，添加警告
    def safe_map(series, mapping):
        unmapped = series[~series.isna() & ~series.isin(mapping.keys())]
        if not unmapped.empty:
            print(f"警告：以下值未在字典中找到映射：{unmapped.unique().tolist()}")
        return series.map(mapping)

    # create new columns with warning check
    new_columns = {
        'trick_part1': safe_map(split_trick[0], word_to_idx),
        'trick_part2': safe_map(split_trick[1], word_to_idx),
        'king_part1': safe_map(split_king[0], word_to_idx),
        'king_part2': safe_map(split_king[1], word_to_idx)
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
    process_file_name = 'data_sample'
    df = load_dataset(os.path.join('dataset', f'{process_file_name}.csv'))
    process_dataset(df)
    df = process_words_columns(df, os.path.join('config', 'word_dict.json'))
    apply_min_max_normalization(df)
    save_dataset(df, os.path.join('dataset', f'processed_{process_file_name}.csv'))

if __name__ == '__main__':
    __main__()
