import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# Manually assign each genre emotional levels
# Source: me. I play a LOT of video games
genre_emotion_map = {
    'Strategy': {'energy': .2, 'tempo': .4, 'valence': .7},
    'Sports': {'energy': .9, 'tempo': .8, 'valence': .4},
    'Simulation': {'energy': .6, 'tempo': .5, 'valence': .4},
    'Shooter': {'energy': .6, 'tempo': .4, 'valence': .9},
    'Role-Playing': {'energy': .7, 'tempo': .8, 'valence': .8},
    'Racing': {'energy': .75, 'tempo': .6, 'valence': .4},
    'Puzzle': {'energy': .1, 'tempo': .1, 'valence': .1},
    'Platform': {'energy': .3, 'tempo': .2, 'valence': .3},
    'Misc': {'energy': .5, 'tempo': .5, 'valence': .5},
    'Fighting': {'energy': .8, 'tempo': .8, 'valence': .8},
    'Adventure': {'energy': .85, 'tempo': .9, 'valence': .8},
    'Action': {'energy': 1.0, 'tempo': 1.0, 'valence': .75}}


def merge_datasets():

    # Get the datasets that I spent oh so much time normalizing and processing or wtv
    games = pd.read_csv('cleaned_data/cleaned_games_dataset.csv')
    song = pd.read_csv('cleaned_data/cleaned_song_dataset.csv')
    audio = pd.read_csv('cleaned_data/cleaned_audio_dataset.csv')

    # We are now going to merge, join 2 datasets together
    merged_dataset_one = pd.merge(games, audio, on='Year', how='outer')
    # Drop any NaN values
    merged_dataset_one = merged_dataset_one.dropna(subset=['Rank'])
    # Delete duplicate Game titles
    merged_dataset_one = merged_dataset_one.drop_duplicates(subset=['Game Title'])
    merged_dataset_one.to_csv('cleaned_data/merged_dataset_one.csv', index=False)

    # Convert the dictionary to a DataFrame
    emotion_mapping_df = pd.DataFrame.from_dict(genre_emotion_map, orient='index')
    emotion_mapping_df.index.name = 'Genre'
    emotion_mapping_df.reset_index(inplace=True)

    # Rename Song column for consistency
    song = song.rename(columns={'popularity': 'Rank_x'})

    # Merge emotional values into the game dataset
    games_mapping_df = games.merge(emotion_mapping_df, on='Genre', how='left')
    games_mapping_df.to_csv('cleaned_data/games_mapped.csv')

    # Create an emotion score by averaging the 3 mappings
    song['emotion_score'] = song[['valence', 'energy', 'tempo']].mean(axis=1)
    games_mapping_df['emotion_score'] = games_mapping_df[['valence', 'energy', 'tempo']].mean(axis=1)

    # Round for more precise matching
    song['emotion_score'] = song['emotion_score'].round(2)
    games_mapping_df['emotion_score'] = games_mapping_df['emotion_score'].round(2)

    # https://medium.com/data-science/how-to-bin-numerical-data-with-pandas-fe5146c9dc55
    # Bucket emotion scores to integers
    song['emotion_bucket'] = (song['emotion_score'] * 100).astype(int)
    games_mapping_df['emotion_bucket'] = (games_mapping_df['emotion_score'] * 100).astype(int)

    # Empty list to store matched chunks
    merged_chunks = []

    # Only match within same or neighboring emotion buckets
    for offset in [-1, 0, 1]:
        shifted_games = games_mapping_df.copy()
        shifted_games['emotion_bucket'] += offset

        temp = pd.merge(song, shifted_games, on='emotion_bucket', suffixes=('_x', '_y'))
        merged_chunks.append(temp)

    # Combine all chunks into one DataFrame
    merged = pd.concat(merged_chunks, ignore_index=True)

    # Compute similarity scores based on emotion and popularity (without normalization)
    merged['emotion_diff'] = abs(merged['emotion_score_x'] - merged['emotion_score_y'])
    merged['popularity_diff'] = abs(merged['Rank_x'] - merged['Rank'])

    # Weigh the similarity score so that emotions are weighed more than popularity
    merged['similarity_score'] = 0.7 * merged['emotion_diff'] + 0.3 * merged['popularity_diff']

    # Select best game for each song
    matched_df = merged.loc[merged.groupby('track_name')['similarity_score'].idxmin()]

    # Drop unneeded columns
    matched_df.drop(columns=[
        'energy_x', 'energy_y', 'tempo_x', 'tempo_y',
        'valence_x', 'valence_y',
        'popularity_norm_x', 'popularity_norm_y',
        'emotion_diff', 'popularity_diff',
        'similarity_score', 'key'
    ], inplace=True, errors='ignore')

    # Save final dataset
    matched_df.to_csv('matched_data.csv', index=False)
    print(matched_df.head(11))


def main():
    merge_datasets()


if __name__ == '__main__':
    main()
