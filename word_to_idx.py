import pandas as pd
import json
# Load the dataset
df = pd.read_csv(r"../odsc-2024-nvidia-hackathon/train.csv")

# Split 'trickortreat' and 'kingofhalloween' columns into two parts each
df[['trick_part1', 'trick_part2']] = df['trickortreat'].str.split('_', expand=True)
df[['king_part1', 'king_part2']] = df['kingofhalloween'].str.split('_', expand=True)

# Combine all unique words from the four new columns
all_words = pd.concat([df['trick_part1'], df['trick_part2'], df['king_part1'], df['king_part2']]).dropna().unique()

# Create a mapping dictionary assigning a unique number to each word
word_to_number = {word: idx for idx, word in enumerate(all_words, start=1)}

# save word mapping to a json file
with open('word_dict.json', 'w') as f:
    json.dump(word_to_number, f)