#https://adventofcode.com/2021/day/6
getwd()
setwd('C:/Users/Anthony/Downloads/AdventOfCode')

tst_input <- '3,4,3,1,2'
myfile <- readLines("input_day06.txt")

get_num_fish <- function(tmp_file, num_days){
  tst_ints <- as.integer(strsplit(tmp_file, ',')[[1]])
  
  tst_output <- tst_ints
  #cat(paste("Initial state:", 
  #            paste0(tst_output, collapse = ',')))
  for(i in 1:num_days){
    if(any(tst_output == 0)){
      n_new <- sum(tst_output==0)
      tst_output[tst_output==0] <- 7
      tst_output <- c(tst_output, rep(9, n_new))
    }
    tst_output <- tst_output - 1
    #cat(paste0("After ", i, " day(s): ", 
    #             paste0(tst_output, collapse = ','), "\n"))
  }
  return(length(tst_output))
}

get_num_fish(tst_input, num_days = 80)
get_num_fish(myfile, num_days = 80)

get_count_fish <- function(tmp_file, num_days){
  tst_ints <- as.integer(strsplit(tmp_file, ',')[[1]])
  
  tst_output <- tst_ints
  counts <- sapply(0:8, function(x) sum(tst_output == x))
  for(i in 1:num_days){
    #move all left (every fish loses one life) and fish on 8 are 0 right now
    new_count <- c(counts[2:9], 0)
    #the fish going from 0 to 6, and the new fish at 8
    new_count[c(7,9)] <- new_count[c(7,9)] + counts[1]
    counts <- new_count
  }
  return(as.character(sum(new_count)))
}

get_count_fish(tst_input, num_days = 256)
get_count_fish(myfile, num_days = 256)

