starts <- c("Richfield" ,"Teevurcher", "Rahora", "Carnsore", "Clooshvalley")

# Base path (assuming Windows)
base_path <- "F:/"
folders <- list.files(base_path)

# Initialize a dataframe to hold the results
results_df <- data.frame(start = character(),
                         data_count = integer(),
                         data2_count = integer(),
                         stringsAsFactors = FALSE)

# Loop through each start
for (start in starts) {
  # Find folders starting with the current start string
  start_folders <- grep(paste0("^", start), folders, value = TRUE)
  
  data_count <- 0
  data2_count <- 0
  
  # Loop through each folder and count files in Data and Data2 subfolders
  for (folder in start_folders) {
    data_path <- file.path(base_path, folder, "Data")
    data2_path <- file.path(base_path, folder, "Data2")
    
    if (file.exists(data_path)) {
      data_files <- list.files(data_path)
      data_count <- data_count + length(data_files)
    }
    
    if (file.exists(data2_path)) {
      data2_files <- list.files(data2_path)
      data2_count <- data2_count + length(data2_files)
    }
  }
  
  # Append the counts to the results dataframe
  results_df <- rbind(results_df, data.frame(start = start, data_count = data_count, data2_count = data2_count))
}

# Print the results dataframe
print(results_df)


df <- read.csv("C:/Users/Anthony/Downloads/model_output_loc_merge.csv")

library(dplyr)
library(ggplot2)
df %>%
  group_by(recorder ,common_name) %>%
  summarize(count = n()) %>%
  ungroup() %>%
  ggplot() + geom_bar(aes(x=common_name, y=count), stat = "identity") +
  facet_wrap(~recorder)

df <- read.csv("C:/Users/ANthony/Downloads/bat_id.csv")
table(df$MANUAL.ID)

bat_species <- c("Brown Long Eared", "Common pipistrelle", 
                 "Myotis sp.", "Nathusius' pipistrelle", 
                 "Nyctalus  leisleri", "Noise", "Nyctalus  leisleri",
                 "Pipistrellus sp.", "Soprano pipistrelle")
df %>%
  filter(`MANUAL.ID` %in% bat_species) %>%
  group_by(MANUAL.ID) %>%
  summarize(count=n()) %>%
  ggplot(aes(x=MANUAL.ID, y = count, label = count)) + geom_bar(stat = "identity") +
  geom_text(vjust=-1) +
  xlab("Bat species")
