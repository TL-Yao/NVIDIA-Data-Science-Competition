import torch
import os
from model_traning import MLP
import pandas as pd

def load_model(input_size):
    model = MLP(input_size=input_size)
    model.load_state_dict(torch.load(os.path.join('.', 'model', 'mlp_model_0_1032.pth')))
    return model

def load_data():
    df = pd.read_csv(os.path.join('.', 'dataset', 'test.csv'))
    return df

def data_preprocessing(df):
    """对测试数据进行预处理，与训练时保持一致"""
    from data_processing import check_negative_one, fill_nan_with_mean, process_words_columns, apply_min_max_normalization
    
    # 按照训练时的顺序进行处理
    check_negative_one(df)
    fill_nan_with_mean(df)
    df = process_words_columns(df, os.path.join('config', 'word_dict.json'))
    df = apply_min_max_normalization(df)
    return df

def inverse_transform_predictions(predictions):
    """将预测结果转换回原始范围"""
    import json
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    # 加载y值的scaler参数
    with open(os.path.join('config', 'y_scaler.json'), 'r', encoding='utf-8') as f:
        y_scaler_params = json.load(f)
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler.min_ = np.array(y_scaler_params["min_"])
        y_scaler.scale_ = np.array(y_scaler_params["scale_"])
        y_scaler.data_min_ = np.array(y_scaler_params["data_min_"])
        y_scaler.data_max_ = np.array(y_scaler_params["data_max_"])
        y_scaler.data_range_ = np.array(y_scaler_params["data_range_"])
    
    # 将预测值转换回原始范围
    predictions_2d = predictions.reshape(-1, 1)
    original_range_predictions = y_scaler.inverse_transform(predictions_2d)
    return original_range_predictions.flatten()

# 预处理测试数据
test_df = load_data()
processed_df = data_preprocessing(test_df)

# 进行预测
model = load_model(input_size=len(processed_df.columns) - 2)  # 减去id和y列
input_data = processed_df.drop(['id', 'y'], axis=1)
predictions = model(torch.tensor(input_data.values, dtype=torch.float32))

# 将预测结果转换回原始范围
original_predictions = inverse_transform_predictions(predictions.detach().numpy())

# 保存预测结果,写入到test.csv的y column，保存在新的文件中
test_df['y'] = original_predictions
test_df = test_df[['id', 'y']]
test_df.to_csv(os.path.join('.', 'dataset', 'test_predictions.csv'), index=False)

