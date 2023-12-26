# MusicTaste Project
This project is divided in part. The first part is related to the EDA (Exploratory Data
Analysis) and ETL (Extract, Transform and Load). Once the database has been modelled according to our preferences, 
features are obtained from the selected tracks.

Then the chosen machine learning (ML) models were trained and the results of which one is the best for each analysis is generated.

### Jamendo Database
The entire database used in this project has been downloaded from [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/).
To be more precise, the complete dataset of mp3 files has been downloaded (raw_30s).

### ETL & EDA
Inside the _EDA_ETL_ folder three scripts and a folder can be found, _etl.R_, _eda.R_,  _reorder.sh_ and _Features_. The first one returns info and charts about the desired genres. It also creates new
tsv files which guarantees that each genre has unique tracks. The second one is where the study of the obtained features is done. Is the final process before training our ML models.
As extra is a script to move the tracks to the desire folder.

#### Feature Extraction
Because our ML models need objective data, a feature extraction is needed. Classification, regression and clustering. The first process is to obtain each feature that could define our target.
Once we have the general dataset, we proceed with the _eda.R_ script and then choose the best features to train the models.

### Machine Learning Models
#### Classification
Considering our aim is to classify a categorical variable as is the genre of a song the selected ML models have been Multiple Logistic Regression, Decision Tree and Random Forest.
For all of them, a cross validation study is done, so the best parameters are chosen.
Everything is in the same loop for, just need to pass the dataframe path and which model you want to train.
#### Regression
As the target is to predict if a song will be liked by the user, the variable to predict will be the song's mark that the user have put to it.
In this case the chosen models have been Ridge, KNN and Decision Tree. In this case cross validatin is used in the last two.
For regression are two functions, one for ridge, jusst need to pass the dataframe path, and for KNN & Decision Tree, which also needs the model selected.