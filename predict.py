import torch
import os
from model_traning import MLP
import pandas as pd

def load_model():
    model = MLP()
    model.load_state_dict(torch.load(os.path.join('.', 'model', 'model.pth')))
    return model

def load_data():
    df = pd.read_csv(os.path.join('.', 'dataset', 'processed_data_sample_xs.csv'))
    return df

def data_preprocessing(df):
    """对测试数据进行预处理，与训练时保持一致"""
    from data_processing import process_dataset, process_words_columns, apply_min_max_normalization
    
    # 按照训练时的顺序进行处理
    process_dataset(df)
    df = process_words_columns(df, os.path.join('config', 'word_dict.json'))
    df = apply_min_max_normalization(df)
    return df

def inverse_transform_predictions(predictions):
    """将预测结果转换回原始范围"""
    import json
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    # 加载y值的scaler参数
    with open(os.path.join('config', 'zero_one_scaler.json'), 'r', encoding='utf-8') as f:
        zero_one_params = json.load(f)
        zero_one_scaler = MinMaxScaler(feature_range=(0, 1))
        zero_one_scaler.min_ = np.array(zero_one_params["min_"])
        zero_one_scaler.scale_ = np.array(zero_one_params["scale_"])
        zero_one_scaler.data_min_ = np.array(zero_one_params["data_min_"])
        zero_one_scaler.data_max_ = np.array(zero_one_params["data_max_"])
        zero_one_scaler.data_range_ = np.array(zero_one_params["data_range_"])
    
    # 将预测值转换回原始范围
    predictions_2d = predictions.reshape(-1, 1)
    original_range_predictions = zero_one_scaler.inverse_transform(predictions_2d)
    return original_range_predictions.flatten()

# 预处理测试数据
test_df = load_data()
processed_df = data_preprocessing(test_df)

# 进行预测
model = load_model()
predictions = model(torch.tensor(processed_df.values, dtype=torch.float32))

# 将预测结果转换回原始范围
original_predictions = inverse_transform_predictions(predictions.detach().numpy())

# 保存预测结果,写入到test.csv的y column，保存在新的文件中
test_df['y'] = original_predictions
test_df = test_df[['id', 'y']]
test_df.to_csv(os.path.join('.', 'dataset', 'test_predictions.csv'), index=False)

