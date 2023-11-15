#Libraries
library(readr)  #read files
library(ggplot2) # charts
library(stringr)
library(dplyr)

# TSV structure
col_struct <- cols(
  track_id <- col_character(),
  artist_id <- col_character(),
  album_id <- col_character(),
  path <- col_character(),
  duration <- col_character(),
)
col_header <- c("TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS")

dfs_names <- list("Pop", "Techno", "Dance", "Alternative", "Rock", "Classical")

for (i in 1:(length(dfs_list))){
  training_path <- paste("EDA_ETL/new_data/",
                         dfs_names[[i]], "_training.tsv", sep = "")
  trainingdf <- read_tsv(training_path,
                         col_names = col_header, col_types = col_struct)
  test_path <- paste("EDA_ETL/new_data/",
                     dfs_names[[i]], "_test.tsv", sep = "")
  testdf <- read_tsv(test_path,
                     col_names = col_header, col_types = col_struct)
  comparationdf <- trainingdf[(trainingdf$TRACK_ID %in% testdf$TRACK_ID), ]
  View(comparationdf)
}