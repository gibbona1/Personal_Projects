getwd()
setwd('C:/Users/Anthony/OneDrive - Maynooth University/Documents/GitHub/Personal_Projects/bird_detection_audio')

today_str <- format(Sys.time(), "%Y%m%d")

today_str <- "20220210"
#install.packages("RcppCNPy")
library(RcppCNPy)


library(stringr)

history_list <- Sys.glob(paste0("model_history/model_history", today_str, "*"))

result_df <- data.frame(run_id   = c(),
                        channels = c(),
                        acc_10   = c())

#history_list

get_row_info <- function(file_str){
  string_run_start <- str_locate(file_str, 'richfield_concat')[,'end'] + 1
  run_id <- as.integer(substr(file_str, string_run_start,string_run_start))
  
  file_end     <- str_locate(file_str, '_.csv')
  
  channels_str <- substr(file_str, string_run_start+1, file_end-1)
  #tail_str <- substr(file_str, str_length('model_history')+1)
  #tail_str <- gsub(".csv", ".npy", tail_str)
  #y_pred <- npyLoad(paste0("y_pred", tail_str), type="integer")
  #y_true <- npyLoad(paste0("y_true", tail_str), type="integer")

  row <- data.frame(run_id   = run_id,
                    channels = channels_str,
                    acc_10   = read.csv(file_str)$val_accuracy[10],
                    loss_10  = read.csv(file_str)$val_loss[10])
  return(row)
}

for(f in history_list){
  row <- get_row_info(f)
  result_df <- rbind(result_df, row)
}


result_df

result_df$col <- ifelse(result_df$channels=='Mod', 'red', 'black')

library(ggplot2)
library(dplyr)
result_df %>%
  filter(grepl("Mod",channels)) %>%
ggplot() + 
  geom_boxplot(aes(channels, acc_10,
               colour = col)) +
  theme(axis.text.x = element_text(size=5)) +
  scale_color_manual(values=c('red'='red', 'black'='black'))+
  ylab("Accuracy after 10 epochs") +
  theme(legend.position = "none")

categories <- c('Common Buzzard',
                'Common Kestrel',
                'Common Snipe',
                'Eurasian Curlew',
                'European Herring Gull',
                'European Robin',
                'Meadow Pipit',
                'Mute Swan',
                'Northern Lapwing',
                'Rook',
                'Tundra Swan',
                'Tundra Swan (Bewicks)')

anova()

one.way <- aov(acc_10 ~ channels, data = result_df)

summary(one.way)
