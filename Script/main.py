
import pandas as pd

from data_cleaning import *
from data_description import *
import kagglehub
from sklearn.preprocessing import StandardScaler
import os  # finds the filepaths for each dataset
import matplotlib
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
    deam_download = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")
    game_download = kagglehub.dataset_download("thedevastator/discovering-hidden-trends-in-global-video-games")
    song_download = kagglehub.dataset_download("ryanholbrook/the-million-songs-dataset")

    # Download latest version of DEAM dataset
    deam_path = "/Users/hayleezuba/.cache/kagglehub/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music/versions/1/DEAM_Annotations/annotations/annotations averaged per song/song_level"
    # print(os.listdir(deam_path)) # shows the files in the directory, for debugging
    deam_df = pd.read_csv(os.path.join(deam_path, 'static_annotations_averaged_songs_2000_2058.csv'))

    # Download latest version of million song subset
    game_path = "/Users/hayleezuba/.cache/kagglehub/datasets/thedevastator/discovering-hidden-trends-in-global-video-games/versions/2"
    # print(os.listdir(game_path))
    games_df = pd.read_csv(os.path.join(game_path, 'Video Games Sales.csv'))

    # Download latest version of Global video game trend subset
    song_path = "/Users/hayleezuba/.cache/kagglehub/datasets/ryanholbrook/the-million-songs-dataset/versions/1"
    # print(os.listdir(song_path))
    songs_df = pd.read_csv(os.path.join(song_path, 'YearPredictionMSD.csv'))

    # Check if the paths are correct and the files exist
    # print("DEAM files:", os.listdir(deam_path))  # Check contents of DEAM dataset folder
    # print("Games files:", os.listdir(game_path))  # Check contents of Games dataset folder
    # print("Songs files:", os.listdir(song_path))  # Check contents of Songs dataset folder

    # Data Preprocessing
    # We will be handling the missing data by dropping rows with missing values, using pandas dropna function
    deam_df = deam_df.dropna()
    games_df = games_df.dropna()
    songs_df = songs_df.dropna()

    # The outliers will be addressed by using the IQR of the data
    deam_df = remove_outliers_IQR(deam_df)
    games_df = remove_outliers_IQR(games_df)
    songs_df = remove_outliers_IQR(songs_df)

    # The data will be normalized using standardization found in SKLearn library
    scaler = StandardScaler()

    # Standardize any numeric columns found in the datasets
    # Strip any leading/trailing whitespace from column names in deam_df
    deam_df.columns = deam_df.columns.str.strip()
    deam_columns = deam_df.select_dtypes(include=['number']).columns
    deam_df[deam_columns] = scaler.fit_transform(deam_df[deam_columns])

    # Strip any leading/trailing whitespace from column names in deam_df
    games_df.columns = games_df.columns.str.strip()
    games_columns = games_df.select_dtypes(include=['number']).columns
    games_df[games_columns] = scaler.fit_transform(games_df[games_columns])

    # Strip any leading/trailing whitespace from column names
    songs_df.columns = songs_df.columns.str.strip()
    song_columns = songs_df.select_dtypes(include=['number']).columns
    songs_df[song_columns] = scaler.fit_transform(songs_df[song_columns])

    # Debugging statement
    print("Preproccessing complete")

    # Perform EDA on DEAM dataset
    print("DEAM dataset: \n")
    print("\nDataset Info:")
    print(deam_df.info())

    print("\nSummary Statistics:")
    print(deam_df.describe())

    # Plot distribution for valence_mean and arousal_mean
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(deam_df['valence_mean'], kde=True, color='purple', bins=20)
    plt.title('Valence Mean Distribution')
    plt.xlabel('Valence Mean')
    plt.ylabel('Frequency')
    plt.savefig('images/deam_valence_distribution.png')  # Save plot to image

    plt.subplot(1, 2, 2)
    sns.histplot(deam_df['arousal_mean'], kde=True, color='grey', bins=20)
    plt.title('Arousal Mean Distribution')
    plt.xlabel('Arousal Mean')
    plt.ylabel('Frequency')
    plt.savefig('images/deam_arousal_distribution.png')  # Save plot to image
    plt.tight_layout()
    plt.show()

    # Plot the correlation between arousal_mean and valence_mean
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='arousal_mean', y='valence_mean', data=deam_df, alpha=0.6, color='purple')
    plt.title('Arousal Mean vs Valence Mean')
    plt.xlabel('Arousal Mean')
    plt.ylabel('Valence Mean')
    plt.grid(True)
    plt.savefig('images/deam_arousal_vs_valence.png')  # Save plot to image
    plt.show()

    # EDA on the Game Trends dataset
    print("\nGame Trends Dataset: \n")
    print("\nDataset Info:")
    print(games_df.info())

    print("\nSummary Statistics:")
    print(games_df.describe())

    # Sale distributions across regions
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Genre', y='Global_Sales', data=games_df, palette='viridis')
    plt.title('Global Sales by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Global Sales (in millions)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('images/games_global_sales_by_genre.png')  # Save plot to image
    plt.show()

    # EDA on the Songs dataset
    print("\nSongs Dataset: \n")
    print("\nDataset Info:")
    print(songs_df.info())

    print("\nSummary Statistics:")
    print(songs_df.describe())

    # Distribution of song figures
    plt.figure(figsize=(12, 6))
    sns.histplot(songs_df['0'], kde=True, color='blue', bins=20)  
    plt.title('Song Feature Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('images/songs_feature_distribution.png')  # Save plot to image
    plt.show()

    # Save images to image file in directory 
    output_dir = 'images'

    # Save plots to the images folder
    plt.savefig(os.path.join(output_dir, 'deam_valence_distribution.png'))  # Save plot to image
    plt.savefig(os.path.join(output_dir, 'deam_arousal_distribution.png'))  # Save plot to image
    plt.savefig(os.path.join(output_dir, 'deam_arousal_vs_valence.png'))  # Save plot to image
    plt.savefig(os.path.join(output_dir, 'games_global_sales_by_genre.png'))  # Save plot to image
    plt.savefig(os.path.join(output_dir, 'songs_feature_distribution.png'))  # Save plot to image


if __name__ == "__main__":
    main()
