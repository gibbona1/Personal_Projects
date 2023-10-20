getwd()
setwd('C:/Users/Anthony/Documents/GitHub/Personal_Projects/bird_detection_audio')

comp_folder <- '20220609_comparison_results'
comp_list   <- list.files(comp_folder)
comp_list
library(stringr)

#new_results_start <- which(str_starts(comp_list, '20220302_16'))[1]
#comp_list <- comp_list[new_results_start:length(comp_list)]

#stringr::str_ends()
metric_data <- c()
for(str_file in comp_list){
  if(str_ends(str_file, 'metric_df.csv'))
    metric_data <- c(metric_data, str_file)
}

metric_data

auc_data <- c()
for(str_file in comp_list){
  if(str_ends(str_file, 'auc_df.csv'))
    auc_data <- c(auc_data, str_file)
}

param_data <- c()
for(str_file in comp_list){
  if(str_ends(str_file, 'param_df.csv'))
    param_data <- c(param_data, str_file)
}

comb_metric_data <- data.frame()
for(metric_file in metric_data){
  df <- read.csv(file.path(comp_folder, metric_file))
  if(nrow(df)>1){
    df$X <- df$X[1]
    df$model <- df$model[df$model != ""][1]
    df <- df[1,]
  }
  comb_metric_data <- rbind(comb_metric_data, df)
}
#comb_metric_data
comb_auc_data <- data.frame()
for(auc_file in auc_data){
  df <- read.csv(file.path(comp_folder, auc_file))
  if(nrow(df)>1){
    df$X <- df$X[1]
    df$model <- df$model[df$model != ""][1]
    df <- df[1,]
  }
  comb_auc_data <- rbind(comb_auc_data, df)
}
read.csv(file.path(comp_folder, auc_file))
auc_data

comb_metric_data

comb_auc_data

comb_param_data <- data.frame()
for(param_file in param_data){
  df <- read.csv(file.path(comp_folder, param_file))
  if(nrow(df)>1){
    df$X <- df$X[1]
    df$model <- df$model[df$model != ""][1]
    df <- df[2,]
  }
  comb_param_data <- rbind(comb_param_data, df)
}

comb_param_data

library(knitr)
#install.packages("kableExtra")
library(kableExtra)
library(dplyr)
library(ggplot2)
library(Rmisc)
comb_metric_data

comb_metric_data[comb_metric_data$model=="smallcnn",]
group_df <- comb_metric_data %>%
  group_by(model) %>%
  dplyr::summarise(avg_acc = mean(top_1_acc), 
                   lci_acc = CI(top_1_acc)[3], 
                   lui_acc = CI(top_1_acc)[1],
                   #avg_acc5 = mean(top_5_acc), 
                   #lci_acc5 = CI(top_5_acc)[3], 
                   #lui_acc5 = CI(top_5_acc)[1],
                   avg_prec = mean(precision), 
                   lci_prec = CI(precision)[3], 
                   lui_prec = CI(precision)[1],
                   avg_f1 = mean(f1), 
                   lci_f1 = CI(f1)[3], 
                   lui_f1 = CI(f1)[1])
group_df
#group_df |> lapply(\(x){ifelse(x == max(x), cell_spec(round(x, 2), bold = TRUE), round(x, 2))}) |> cbind(df2) |> 
group_df %>%
  mutate(across(c(avg_acc, avg_prec, avg_f1), ~cell_spec(., bold = . == max(.)))) %>%
  kable(escape = FALSE, booktabs = TRUE, digits=3,
        col.names = c("model", rep(c("mean", "Lower CI", "Upper CI"), 3))) %>%
  kable_styling(
    full_width = FALSE,
    bootstrap_options = c("striped", "hover", "condensed"), 
  ) %>%
  add_header_above(c( '', Accuracy1 = 3, Precision = 3, F1 = 3)) 

library(ggplot2)
comb_metric_data %>%
  #filter(model %in% c('resnet50', 'resnet50_concat', 'vgg19', 'vgg19_concat')) %>%
ggplot() + 
  geom_boxplot(aes(model, top_1_acc)) +
  #theme(axis.text.x = element_text(size=5)) +
  #scale_color_manual(values=c('red'='red', 'black'='black'))+
  ylab("Test Accuracy") +
  theme(legend.position = "none")

ggplot(comb_auc_data) + 
  geom_boxplot(aes(model, auc_CommonKestrel)) +
  #theme(axis.text.x = element_text(size=5)) +
  #scale_color_manual(values=c('red'='red', 'black'='black'))+
  ylab("Best AUC") +
  theme(legend.position = "none")

comb_metric_data

library(dplyr)
#install.packages("Rmisc")
library(Rmisc)

#comb_metric_data$X <- rep(1:5, each=7)
comb_metric_data %>%
  mutate(model = model %>% as.factor()) %>%
  group_by(model) %>%
  dplyr::summarise(avg_acc = mean(top_1_acc), 
            lci_acc = CI(top_1_acc)[3], 
            lui_acc = CI(top_1_acc)[1])


mtcars %>% t %>%
  as.data.frame %>%
  mutate(across(everything(), ~ cell_spec(., bold = . == max(.)))) %>%
  t %>%
  as.data.frame %>%
  kable(escape = FALSE, booktabs = TRUE) %>%
  
  kable_styling(
    full_width = FALSE,
    bootstrap_options = c("striped", "hover", "condensed"), 
  ) 
