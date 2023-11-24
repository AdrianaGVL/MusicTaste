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
df <- read.csv(file.path(df_path, "df.csv"),
               sep = ";", dec = ",", header = TRUE)

# Visualization data types
str(df)
tipos_de_datos <- sapply(df, class)
print(tipos_de_datos)

# Change some notation
for (col in colnames(df)) {
  new_col_name <- gsub("\\.", "_", col)
  colnames(df)[colnames(df) == col] <- new_col_name
}
names(df)[names(df) == "Energy  dB "] <- "Energy"
names(df)[names(df) == "Loudness  dB "] <- "Loudness"

# Transform data to numeric format
features_to_trans <- setdiff(names(df), c("Genre"))
df <- df %>%
  mutate_at(vars(features_to_trans), ~as.numeric(as.character(.)))
# Visualization data types
str(df)
tipos_de_datos <- sapply(df, class)
print(tipos_de_datos)

# Features boxplot
feats <- summary(select(df, -Genre))
print(feats)
cols <- names(df)
no_num <- "Genre"
features <- cols[!(cols %in% no_num)]
for (feat in features) {
  filename <- paste("EDA_ETL/Features/new_data/boxplot_", gsub(" ", "_", feat), ".png",sep = "")
  png(filename, width = 800, height = 600, units = "px", pointsize = 12)
  boxplot(df[[feat]], main = paste(feat, "Boxplot"), ylab = "Values")
  dev.off()
}

#Plot scatter matrices between features
feats_freq <- setdiff(names(df), c('Genre','Tempo', 'Beats per song', 'Danceability', 'Loudness', 'Energy', 'Spectral Rolloff', 'Spectral Centroid'))
feats_not_freq <- setdiff(names(df), c('Genre', 'MFCC 1 mean', 'MFCC 1 min',' MFCC 1 max', 'MFCC 10 mean', 'MFCC 10 min', 'MFCC 10 max', 'MFCC 20 mean', 'MFCC 20 min', 'MFCC 20 max'))
scatter_matrix <- ggpairs(select(df, all_of(features_to_trans)),
                                 diag = list(continuous = "densityDiag"),
                                 upper = list(continuous = wrap("cor", digits = 2)),
                                 axisLabels = "none")
scatter_matrix <- scatter_matrix <- scatter_matrix + theme(axis.text.x = element_text(angle = 45, hjust = 1))
filename <- "EDA_ETL/Features/new_data/scatter_matrix_all.png"
ggsave(filename, scatter_matrix, width = 12, height = 10, units = "in", dpi = 300)


# Correlation between features
subset_df <- df[, feats_not_freq]
correlation_matrix <- cor(subset_df)
corrplot(correlation_matrix, method = "color", tl.col = "black", tl.srt = 45)
filename <- "EDA_ETL/Features/new_data/correlation_not_freq.png"
png(filename, width = 800, height = 600, units = "px", pointsize = 12)
corrplot(correlation_matrix, method = "color", tl.col = "black", tl.srt = 45)
dev.off()


# Relation Between features and genres - ANOVA
not_categ <- names(df)[2:ncol(df)]
anova_study <- lapply(not_categ, function(f){
  formula <- as.formula(paste(f, "~ Genre"))
  anova <- aov(formula, data = df)
  tidy_result <- tidy(anova)
  tidy_result$Variable <- f
  return(tidy_result)
})
results_table <- do.call(rbind, anova_study)
write.xlsx(results_table, file = "EDA_ETL/Features/new_data/ANOVA_results.xlsx", rowNames = FALSE)