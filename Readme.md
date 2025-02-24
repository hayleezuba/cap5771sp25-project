# CAP5771 Spring 2025 Project, Haylee Zuba

# Replicating the Findings
In order to replicate the EDA I performed on the data, you first must get the datasets from kaggle using the statements below. This code is using a subset of the million song dataset because running EDA on a 300GB database is extensive, and by taking a random subset, it is possible to identify certain trends within the data. 

deam_path = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")

game_path = kagglehub.dataset_download("thedevastator/discovering-hidden-trends-in-global-video-games")

song_path = kagglehub.dataset_download("ryanholbrook/the-million-songs-dataset")

After you have the data, you must perform the EDA code on it. By running the main, the pictures take a very long time to graph. I need to go in and fix this but unfortunately, procrastination has been my enemy this semester. 
