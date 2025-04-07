library(dplyr)
library(tidyr)
library(stringr)
# Load reticulate and import numpy
library(reticulate)

df <- read.csv("../../Downloads/neal_labels - Copy.csv")

np <- import("numpy")

# Load the .npz file (replace 'your_file.npz' with the actual file path)
npz_file <- np$load("../../Downloads/specdata.npz", allow_pickle = TRUE)

class_names <- npz_file$get("categories")

get_prop_in_clases <- function(df)
  return(df$class_label %in% class_names %>% mean())

get_prop_in_clases(df) * nrow(df)

df %>%
  mutate(class_label = str_trim(class_label)) %>%
  mutate(class_label = str_to_title(class_label)) %>%
  mutate(class_label = str_replace_all(class_label, "Wren", "Eurasian Wren")) %>%
  mutate(class_label = str_replace_all(class_label, "Blackbird", "Eurasian Blackbird")) %>%
  mutate(class_label = str_replace_all(class_label, "Linnet", "Eurasian Linnet")) %>%
  mutate(class_label = str_replace_all(class_label, "Stonechat", "European Stonechat")) %>%
  mutate(class_label = str_replace_all(class_label, "Goldfinch", "European Goldfinch")) %>%
  mutate(class_label = str_replace_all(class_label, "European Robin", "Robin")) %>%
  mutate(class_label = str_replace_all(class_label, "Robin", "European Robin")) %>%
  mutate(class_label = str_replace_all(class_label, "Chaffinch", "Common Chaffinch")) %>%
  mutate(class_label = str_replace_all(class_label, "Blue Tit", "Eurasian Blue Tit")) %>%
  mutate(class_label = str_replace_all(class_label, "Woodpigeon", "Common Wood-Pigeon")) %>%
  mutate(class_label = str_replace_all(class_label, "Greattit", "Great Tit")) %>%
  mutate(class_label = str_replace_all(class_label, "Skylark", "Eurasian Skylark")) %>%
  
  #pull(class_label) %>% unique() %>% sort()
  #write.csv("../../Downloads/neal_labels_remapped.csv")
  get_prop_in_clases() * nrow(df)
  

df$class_label
