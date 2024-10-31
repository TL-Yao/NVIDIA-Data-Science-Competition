import pandas as pd
import json
import os
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import MinMaxScaler
def load_dataset(path):
    # Load the dataset
    df = pd.read_csv(path)
    return df

def primary_component_analysis(df, file_name):
    # 分离出id和y列
    id_col = df['id'] if 'id' in df.columns else None
    y_col = df['y'] if 'y' in df.columns else None
    
    # 获取要进行PCA的特征列
    feature_cols = [col for col in df.columns if col not in ['id', 'y']]
    features_df = df[feature_cols]
    
    save_pca = False

    # if pca model already exists, load it
    if os.path.exists(os.path.join('.', 'config', 'pca_model.pkl')):
        with open(os.path.join('.', 'config', 'pca_model.pkl'), 'rb') as f:
            pca = pickle.load(f)
    else:
        pca = PCA(n_components=0.95)
        pca.fit(features_df)
        print(f"Number of components to keep 95% of the variance: {pca.n_components_}")
        save_pca = True
        
    # transform the data
    df_pca = pca.transform(features_df)
    
    if save_pca:
        # save pca model
        with open(os.path.join('.', 'config', 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f)
    
    # normalize pca data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_pca = scaler.fit_transform(df_pca)

    # 将转换后的数据转换为DataFrame
    df_pca = pd.DataFrame(df_pca)
    
    # 如果存在id和y列,将它们添加回来
    if id_col is not None:
        df_pca.insert(0, 'id', id_col)
    if y_col is not None:
        df_pca['y'] = y_col

    # save transformed data into csv file
    df_pca.to_csv(os.path.join('.', 'dataset', f'pca_{file_name}.csv'), index=False)
    


def __main__():
    file_name = 'processed_data_sample'
    df = load_dataset(os.path.join('.', 'dataset', f'{file_name}.csv'))
    primary_component_analysis(df, file_name)

if __name__ == "__main__":
    __main__()


