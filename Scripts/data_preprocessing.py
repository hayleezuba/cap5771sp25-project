import kagglehub
from sklearn.preprocessing import StandardScaler
import os  # finds the filepaths for each dataset
import matplotlib
from data_manipulation import *
matplotlib.use('TkAgg')


# Remove outliers based on the IQR of the dataset
def remove_outliers_IQR(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        # For each column, obtain the IQR by computing the Q1, Q3 and uper and lower bounds
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Only keep the values within the IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def main():
    # Load in the datasets
    game_download = kagglehub.dataset_download("thedevastator/discovering-hidden-trends-in-global-video-games")
    audio_features_download = kagglehub.dataset_download("uciml/msd-audio-features")
    song_features_download = kagglehub.dataset_download("nanditapore/spotify-track-features")

    # Download latest version of Video game sales dataset
    game_path = "/Users/hayleezuba/.cache/kagglehub/datasets/thedevastator/discovering-hidden-trends-in-global-video-games/versions/2"
    # print(os.listdir(game_path))
    # I only want certain columns from this dataset, so Im going to choose them manually
    games_df = pd.read_csv(os.path.join(game_path, 'Video Games Sales.csv'), usecols=['Game Title', 'Rank', 'Genre', 'Year'])

    # Download latest version of Audio Feature dataset
    audio_path = "/Users/hayleezuba/.cache/kagglehub/datasets/uciml/msd-audio-features/versions/1"
    # print(os.listdir(audio_path))
    audio_df = pd.read_csv(os.path.join(audio_path, 'year_prediction.csv'), usecols=['label', 'TimbreAvg1'])

    # Download new song dataset
    spotify_song_path = "/Users/hayleezuba/.cache/kagglehub/datasets/nanditapore/spotify-track-features/versions/1"
    # print(os.listdir(spotify_song_path))
    song_df = pd.read_csv(os.path.join(spotify_song_path, 'song_track.csv'), usecols=['track_name', 'genre',
                                                                                      'artist_name', 'popularity',
                                                                                      'energy', 'tempo', 'valence'])

    # Save raw data
    os.makedirs('raw_data', exist_ok=True)  # Create directory if it doesn't exist
    games_df.to_csv('raw_data/games_dataset.csv', index=False)
    audio_df.to_csv('raw_data/audio_dataset.csv', index=False)
    song_df.to_csv('raw_data/song_dataset.csv', index=False)

    # Data Preprocessing
    # We will be handling the missing data by dropping rows with missing values, using pandas dropna function
    games_df = games_df.dropna()
    audio_df = audio_df.dropna()
    song_df = song_df.dropna()
    # Rename audio column bc I dont need that name
    audio_df.rename(columns={'label': 'Year'}, inplace=True)

    # The outliers will be addressed by using the IQR of the data
    games_df = remove_outliers_IQR(games_df)
    audio_df = remove_outliers_IQR(audio_df)
    song_df = remove_outliers_IQR(song_df)

    # The data will be normalized using standardization found in SKLearn library
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize any numeric columns found in the datasets
    # I don't want to standardize the whole datasets with themselves, so I only want to standardize certain columns

    # Video Game normalization
    games_df['Rank'] = games_df['Rank'].max() - games_df['Rank']  # So its not inverted
    games_df[['Rank']] = scaler.fit_transform(games_df[['Rank']])

    # Audio feature normalization
    audio_df['TimbreAvg1'] = audio_df['TimbreAvg1'].max() - audio_df['TimbreAvg1']
    audio_df[['TimbreAvg1']] = scaler.fit_transform(audio_df[['TimbreAvg1']])

    # Song feature normalization
    # Normalize them individually
    numerical_columns = ['popularity', 'energy', 'tempo', 'valence']

    for col in numerical_columns:
        # Invert the columns
        song_df[col] = song_df[col].max() - song_df[col]
        # Normalize
        song_df[col] = scaler.fit_transform(song_df[[col]])

    # Save new datasets
    os.makedirs('cleaned_data', exist_ok=True)  # Create directory if it doesn't exist
    games_df.to_csv('cleaned_data/cleaned_games_dataset.csv', index=False)
    audio_df.to_csv('cleaned_data/cleaned_audio_dataset.csv', index=False)
    song_df.to_csv('cleaned_data/cleaned_song_dataset.csv', index=False)

    # Debugging statement
    print("Preprocessing complete")


if __name__ == "__main__":
    main()
