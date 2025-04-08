from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def feature_engineering():
    # Get the merged csv
    df = pd.read_csv('matched_data.csv')

    # Label encode the categorical columns
    for col in ['artist_name', 'track_name', 'Game Title', 'Genre', 'genre']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Create the engagement_score from popularity and ranks
    df['engagement_score'] = (df['Rank_x'] * 0.5 + df['Rank'] * 0.5).round(3)

    # Save encoded data
    df.to_csv('encoded_matched_data.csv', index=False)

    # Train/test split
    y = df['Game Title']
    X = df.drop(columns=['Game Title'])

    # Split and export
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

def main():
    feature_engineering()

if __name__ == '__main__':
    main()
