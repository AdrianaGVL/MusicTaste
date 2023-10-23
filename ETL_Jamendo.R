###############################
# Author: Adriana Galán
# Music Taste Project
###############################

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

# ######################################
# # JUST IN CASE TSV FILES HAS \t or " "
# tsv_path <- "original_file.tsv"
# df <- read_tsv(tsv_path, col_names = col_header, col_types = col_struct)
# df$TAGS <- gsub("[\r\n]+|\\s+", ", ", df$TAGS) # Changes \t and " " to ", "
# write.table(df, file = "clean_file.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
# #######################################


# TSV path
tsv_path <- "/Users/agv/Estudios/Universidad/Máster/PRDL+MLLB/mtg-jamendo-dataset/data/clean_agv/raw_30s_comas.tsv"

# Read tsv and generate a dataframe
df <- read_tsv(tsv_path, col_names = col_header, col_types = col_struct)

## ETL
# NaN or NULL cells
celdas_vacias <- sum(is.na(df)) + sum(df == "")
if (celdas_vacias > 0) {
  cat("There are", celdas_vacias, ".\n")
} else {
  cat("TSV file has any null cell.\n")
}

# Duplicate values
duplicates <- duplicated(df$TRACK_ID)
dvalues <- df$TRACK_ID[duplicates]
count_dvalues <- sum(duplicates)

# Sub dataframes, one per chosen genre
pop_df <- df[str_detect(df$TAGS, "\\genre---pop\\b"), ]
techno_df <- df[str_detect(df$TAGS, "\\genre---techno\\b"), ]
dance_df <- df[str_detect(df$TAGS, "\\genre---dance\\b"), ]
alternative_df <- df[str_detect(df$TAGS, "\\genre---alternative\\b"), ]
rock_df <- df[str_detect(df$TAGS, "\\genre---rock\\b"), ]
classical_df <- df[str_detect(df$TAGS, "\\genre---classical\\b"), ]

# Remove repeated tracks from the genres with most tracks
rows_remove <- function(datafr, colum, value_to_match) {
  track_row <- which(colum == value_to_match)
  aux_datafr <- datafr[-track_row, , drop = FALSE]
  return(aux_datafr)
}

dfs_list <- list(pop_df, techno_df,
                 dance_df, alternative_df,
                 rock_df, classical_df)
dfs_names <- list("Pop", "Techno", "Dance", "Alternative", "Rock", "Classical")

# Chart with songs per genre
genres_df <- data.frame(DataFrame = unlist(dfs_names),
                        CommonIDs = sapply(dfs_list, function(df) length(unique(df$TRACK_ID))))
g <- ggplot(genres_df, aes(x = DataFrame, y = CommonIDs, fill = DataFrame)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = CommonIDs), vjust = -0.5, size = 4) +
  theme_minimal() +
  labs(
    title = "Quantity of tracks per genre",
    x = "Genres",
    y = "Quantity of tracks in common"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "none") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Songs_per_Genre.png", plot = g, width = 8, height = 6, units = "in")


# # Compare one many songs from each genre are also in other genres
# ## All of them
# ids_list <- lapply(dfs_list, function(df) unique(df$TRACK_ID))
# common_ids <- Reduce(intersect, ids_list)
# # cat("The quantity of tracks in common is:", length(common_ids), "\n")

## Four by four
in_common_counts <- list()
in_common_list_4by4 <- list()
for (i in 1:(length(dfs_list) - 3)) {
  for (j in (i + 1):(length(dfs_list) - 2)) {
    for (k in (j + 1):(length(dfs_list) - 1)) {
      for (l in (k + 1):length(dfs_list)) {
        if (i != j && i != k && i != l && j != k && j != l && k != l) {
          id1 <- unique(dfs_list[[i]]$TRACK_ID)
          id2 <- unique(dfs_list[[j]]$TRACK_ID)
          id3 <- unique(dfs_list[[k]]$TRACK_ID)
          id4 <- unique(dfs_list[[l]]$TRACK_ID)
          ids_in_common <- Reduce(intersect, list(id1, id2, id3, id4))
          in_common_count <- length(ids_in_common)
          if (in_common_count > 0) {
            comparation <- paste("From ", dfs_names[i], ", ",
                                 dfs_names[j], ", ", dfs_names[k], " & ",
                                 dfs_names[l], " lists:", sep = "")
            in_common_list_4by4[[comparation]] <- ids_in_common
            cat(comparation,
                "The quantity of tracks in common is:",
                length(ids_in_common), "\n")

            num_rows <- list(length(id1), length(id2), length(id3), length(id4))
            index_fewer_rows <- which.min(numRows)
            for (t in 1:in_common_count) {
              if (index_fewer_rows == 1) {
                dfs_list[[j]] <- rows_remove(dfs_list[[j]],
                                             dfs_list[[j]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[k]] <- rows_remove(dfs_list[[k]],
                                             dfs_list[[k]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[l]] <- rows_remove(dfs_list[[l]],
                                             dfs_list[[l]]$TRACK_ID,
                                             ids_in_common[t])
              }
              if (index_fewer_rows == 2) {
                dfs_list[[i]] <- rows_remove(dfs_list[[i]],
                                             dfs_list[[i]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[k]] <- rows_remove(dfs_list[[k]],
                                             dfs_list[[k]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[l]] <- rows_remove(dfs_list[[l]],
                                             dfs_list[[l]]$TRACK_ID,
                                             ids_in_common[t])
              }
              if (index_fewer_rows == 3) {
                dfs_list[[i]] <- rows_remove(dfs_list[[i]],
                                             dfs_list[[i]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[j]] <- rows_remove(dfs_list[[j]],
                                             dfs_list[[j]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[l]] <- rows_remove(dfs_list[[l]],
                                             dfs_list[[l]]$TRACK_ID,
                                             ids_in_common[t])
              }
              if (index_fewer_rows == 4) {
                dfs_list[[i]] <- rows_remove(dfs_list[[i]],
                                             dfs_list[[i]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[j]] <- rows_remove(dfs_list[[j]],
                                             dfs_list[[j]]$TRACK_ID,
                                             ids_in_common[t])
                dfs_list[[k]] <- rows_remove(dfs_list[[k]],
                                             dfs_list[[k]]$TRACK_ID,
                                             ids_in_common[t])
              }
            }
          }
        }
      }
    }
  }
}

## Three by three
in_common_counts <- list()
in_common_list_3by3 <- list()

combis <- combn(dfs_names, 3)
function_tag_format <- function(combis) {
  paste(combis, collapse = " & ")
}
comas_combinations <- apply(combis, 2, function_tag_format)
combinations <- sub(" & ", ", ", comas_combinations, fixed = TRUE)

for (i in 1:(length(dfs_list) - 2)) {
  for (j in (i + 1):length(dfs_list) - 1) {
    for (k in (i + 2):length(dfs_list)){
      if (i != j && j != k && i != k) {
        id1 <- unique(dfs_list[[i]]$TRACK_ID)
        id2 <- unique(dfs_list[[j]]$TRACK_ID)
        id3 <- unique(dfs_list[[k]]$TRACK_ID)
        ids_in_common <- Reduce(intersect, list(id1, id2, id3))
        in_common_count <- length(ids_in_common)
        if (in_common_count > 0) {
          trio_tag <- paste(dfs_names[i], ", ",
                            dfs_names[j], " & ",
                            dfs_names[k], sep = "")
          if (trio_tag %in% combinations) {
            in_common_counts[[trio_tag]] <- in_common_count
          }
          in_common_list_3by3[[trio_tag]] <- ids_in_common

          num_rows <- list(length(id1), length(id2), length(id3))
          index_fewer_rows <- which.min(numRows)
          for (t in 1:in_common_count) {
            if (index_fewer_rows == 1) {
              dfs_list[[j]] <- rows_remove(dfs_list[[j]],
                                           dfs_list[[j]]$TRACK_ID,
                                           ids_in_common[t])
              dfs_list[[k]] <- rows_remove(dfs_list[[k]],
                                           dfs_list[[k]]$TRACK_ID,
                                           ids_in_common[t])
            }
            if (index_fewer_rows == 2) {
              dfs_list[[i]] <- rows_remove(dfs_list[[i]],
                                           dfs_list[[i]]$TRACK_ID,
                                           ids_in_common[t])
              dfs_list[[k]] <- rows_remove(dfs_list[[k]],
                                           dfs_list[[k]]$TRACK_ID,
                                           ids_in_common[t])
            }
            if (index_fewer_rows == 3) {
              dfs_list[[i]] <- rows_remove(dfs_list[[i]],
                                           dfs_list[[i]]$TRACK_ID,
                                           ids_in_common[t])
              dfs_list[[j]] <- rows_remove(dfs_list[[j]],
                                           dfs_list[[j]]$TRACK_ID,
                                           ids_in_common[t])
            }
          }
        }
      }
    }
  }
}
# Chart
o <- data.frame(trios = names(in_common_counts),
                count = unlist(in_common_counts))
g3by3 <- ggplot(o, aes(x = trios, y = count, fill = trios)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = count), vjust = -0.5, size = 4) +
  theme_minimal() +
  labs(
    title = "Tracks in common",
    x = "Genres trios with tracks in common",
    y = "Quantity of tracks in common"
  ) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  theme(legend.position = "right") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("tracks_in_common_3by3.png",
       plot = g3by3, width = 8, height = 6, units = "in")



## Two by Two
in_common_counts <- list()
in_common_list_2by2 <- list()
for (i in 1:(length(dfs_list) - 1)) {
  for (j in (i + 1):length(dfs_list)) {
    id1 <- unique(dfs_list[[i]]$TRACK_ID)
    id2 <- unique(dfs_list[[j]]$TRACK_ID)
    ids_in_common <- intersect(id1, id2)
    in_common_count <- length(ids_in_common)
    pair_tag <- paste(dfs_names[i], " & ", dfs_names[j])
    in_common_list_2by2[[pair_tag]] <- ids_in_common
    in_common_counts[[pair_tag]] <- in_common_count
    if (in_common_count > 0) {
      for (t in 1:in_common_count) {
        if (index_fewer_rows == 1) {
          dfs_list[[j]] <- rows_remove(dfs_list[[j]],
                                       dfs_list[[j]]$TRACK_ID,
                                       ids_in_common[t])
        }
        if (index_fewer_rows == 2) {
          dfs_list[[i]] <- rows_remove(dfs_list[[i]],
                                       dfs_list[[i]]$TRACK_ID,
                                       ids_in_common[t])
        }
      }
    }
  }
}
# Chart
p <- data.frame(pairs = names(in_common_counts),
                count = unlist(in_common_counts))
g2by2 <- ggplot(p, aes(x = pairs, y = count, fill = pairs)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = count), vjust = -0.5, size = 4) +
  theme_minimal() +
  labs(
    title = "Tracks in common",
    x = "Genres pairs with tracks in common",
    y = "Quantity of tracks in common"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "none") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("tracks_in_common_2by2_cleaning.png",
       plot = g2by2, width = 8, height = 6, units = "in")



# Chart with songs per genre
genres_df <- data.frame(DataFrame = unlist(dfs_names),
                        CommonIDs = sapply(dfs_list, function(df) length(unique(df$TRACK_ID))))
g <- ggplot(genres_df, aes(x = DataFrame, y = CommonIDs, fill = DataFrame)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = CommonIDs), vjust = -0.5, size = 4) +
  theme_minimal() +
  labs(
    title = "Quantity of tracks per genre",
    x = "Genres",
    y = "Quantity of tracks in common"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "none") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Songs_per_Genre_cleaning.png",
       plot = g, width = 8, height = 6, units = "in")