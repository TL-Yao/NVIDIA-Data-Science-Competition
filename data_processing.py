import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json

# Load the dataset
df = pd.read_csv(r"../odsc-2024-nvidia-hackathon/train.csv")

# Fill NaN values in each column with the mean of that column
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:  
        df[column].fillna(df[column].mean(), inplace=True)

# Choose a column to visualize
chosen_column = df.select_dtypes(include=['float64', 'int64']).columns.difference(['id', 'Y', 'trickortreat', 'kingofhalloween'])[0]

# Apply Min-Max normalization and replace the original column
scaler = MinMaxScaler()
df[chosen_column] = scaler.fit_transform(df[[chosen_column]])

# Split 'trickortreat' and 'kingofhalloween' columns into two parts each
df[['trick_part1_temp', 'trick_part2_temp']] = df['trickortreat'].str.split('_', expand=True)
df[['king_part1_temp', 'king_part2_temp']] = df['kingofhalloween'].str.split('_', expand=True)

word_to_idx = json.load(open('../odsc-2024-nvidia-hackathon/word_dict.json', 'r'))

# insert word to idx to the dataframe to replace thrid and fourth column
df.insert(3, 'trick_part1', df['trick_part1_temp'].map(word_to_idx))
df.insert(4, 'trick_part2', df['trick_part2_temp'].map(word_to_idx))
df.insert(5, 'king_part1', df['king_part1_temp'].map(word_to_idx))
df.insert(6, 'king_part2', df['king_part2_temp'].map(word_to_idx))

# remove the trickortreat and kingofhalloween columns
df.drop(columns=['trickortreat', 'kingofhalloween', 'trick_part1_temp', 'trick_part2_temp', 'king_part1_temp', 'king_part2_temp'], inplace=True)

# fill the nan values in the 3-6 columns with mode
for i in range(2, 6):
    # print the mode of the column
    print(df[df.columns[i]].mode())
    df[df.columns[i]].fillna(df[df.columns[i]].mode()[0], inplace=True)

# save the dataframe to a csv file
df.to_csv('../odsc-2024-nvidia-hackathon/processed_train.csv', index=False)
