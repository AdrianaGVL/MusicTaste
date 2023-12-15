##############################
#   Author:: Adriana Galán
#   Music Taste Project
############################

# Libraries
library(tidyverse)
library(GGally)
library(corrplot)
library(dplyr)
library(Hmisc)
library(psych)
library(polycor)
library(ltm)
library(reshape2)
library(broom)
library(sjPlot)
library(flextable)
library(stats)
library(openxlsx)


# Import data
df_path <- file.path("EDA_ETL/Features/new_data")
df <- read.csv(file.path(df_path, "df_marks.csv"),
               sep = ";", dec = ",", header = TRUE)

# # Visualization data types
# str(df)
# data_types <- sapply(df, class)
# print(data_types)

# Change some notation
for (col in colnames(df)) {
  new_col_name <- gsub("\\.", "_", col)
  colnames(df)[colnames(df) == col] <- new_col_name
}

# # Transform data to numeric format
features_to_trans <- setdiff(names(df), c("Genre"))
df <- df %>%
  mutate_at(vars(features_to_trans), ~as.numeric(as.character(.)))
# # Visualization data types
# str(df)
# data_types <- sapply(df, class)
# print(data_types)

# Features boxplot
# feats <- summary(select(df, -Genre))
# print(feats)
# cols <- names(df)
# no_num <- "Genre"
# features <- cols[!(cols %in% no_num)]
# for (feat in features) {
#   filename <- paste("EDA_ETL/Features/new_data/boxplot_", gsub(" ", "_", feat), ".png",sep = "")
#   png(filename, width = 800, height = 600, units = "px", pointsize = 12)
#   boxplot(df[[feat]], main = paste(feat, "Boxplot"), ylab = "Values")
#   dev.off()
# }

# # #Plot scatter matrices between features
# freq_features <- character(20)  # Crear un vector de caracteres vacío con longitud 20
# for (num in 1:20) {
#   freq_features[num] <- paste("MFCC", num, sep = "_")
# }
# feats_not_freq <- setdiff(names(df), freq_features)
# feats_freq <- setdiff(names(df), c('Genre', 'Beats_song', 'Danceability', 'Loudness', 'Spectral_Rolloff', 'Spectral Centroid'))
# # feats_not_freq <- setdiff(names(df), c('Genre', 'MFCC 1', 'MFCC 2','MFCC 3', 'MFCC 4', 'MFCC 10 min', 'MFCC 10 max', 'MFCC 20 mean', 'MFCC 20 min', 'MFCC 20 max'))
# scatter_matrix <- ggpairs(select(df, all_of(feats_freq)),
#                                  diag = list(continuous = "densityDiag"),
#                                  upper = list(continuous = wrap("cor", digits = 2)),
#                                  axisLabels = "none")
# scatter_matrix <- scatter_matrix <- scatter_matrix + theme(axis.text.x = element_text(angle = 45, hjust = 1))
# filename <- "EDA_ETL/Features/new_data/scatter_matrix_freq_MFCC_MARK.png"
# ggsave(filename, scatter_matrix, width = 12, height = 10, units = "in", dpi = 300)


# # Correlation between features
# subset_df <- df[, feats_not_freq]
# correlation_matrix <- cor(subset_df)
# corrplot(correlation_matrix, method = "color", tl.col = "black", tl.srt = 45)
# filename <- "EDA_ETL/Features/new_data/correlation_Nootfreq_MARK.png"
# png(filename, width = 800, height = 600, units = "px", pointsize = 12)
# corrplot(correlation_matrix, method = "color", tl.col = "black", tl.srt = 45)
# dev.off()


# # Relation Between features and genres - ANOVA
# # Normalisation
# for (feat in features_to_trans) {
#   df[[feat]] <- scale(df[[feat]])
# }
# # ANOVA
# not_categ <- names(df)[7:ncol(df)]
# anova_study <- lapply(not_categ, function(f){
#   formula <- as.formula(paste(f, "~ Mark"))
#   anova <- aov(formula, data = df)
#   tidy_result <- tidy(anova)
#   print(tidy_result)
#   tidy_result$Variable <- f
#   return(tidy_result)
# })
# results_table <- do.call(rbind, anova_study)
# write.xlsx(results_table, file = "EDA_ETL/Features/new_data/ANOVA_Marks_results.xlsx", rowNames = FALSE)