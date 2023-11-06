# MusicTaste Project
This project is divided in part, as it has grown. of course, the first part is related to the EDA (Exploratory Data
Analysis) and ETL (Extract, Transform and Load). Once the database has been modelled according to our preferences, 
features are obtained from the selected tracks.

### Jamendo Database
The entire database used in this project has been downloaded from [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/).
To be more precise, the complete dataset of mp3 files has been downloaded (raw_30s).

### ETL (Extract, Transform and Load)
Inside the _EDA_ETL_ folder to scripts can be found, _ETL_Jamendo.R_ (this name is because the downloaded database and
already commented), and reorder.sh. The first one returns info and charts about the desired genres. It also creates new
tsv files which guarantees that each genre has unique tracks.

Once the selected groups of music files are selected, they can be reorganised into new folders with reorder.sh script.

### Feature Extraction
