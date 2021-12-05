#https://adventofcode.com/2021/day/3

getwd()
setwd('C:/Users/Anthony/Downloads/AdventOfCode')
#myfile <- read.table('input_day03.txt')
scan_zero <- do.call(rbind, strsplit(readLines("input_day03.txt"), ""))

library(data.table)

tst_file <- c('00100', '11110', '10110', '10111', '10101', '01111', '00111', 
  '11100', '10000', '11001', '00010', '01010')
tst_file <- do.call(rbind, strsplit(tst_file, ""))

get_ox_rate <- function(df){
  getmode <- function(v) {
    #so the mode is 1 when 0 and 1 appear an equal amount of time
    uniqv <- rev(unique(v))
    as.integer(uniqv[which.max(tabulate(match(v, uniqv)))])
  }
  
  n <- ncol(df)
  mode_vec <- apply(df, 2, getmode)
  bin_vec  <- 2^((n-1):0)
  return(sum(mode_vec * bin_vec)*sum((1-mode_vec) * bin_vec)
)
}

get_ox_rate(tst_file)
get_ox_rate(scan_zero)

get_crit2 <- function(df, mode = "ox"){
  df_copy <- df
  for(cx in 1:ncol(df)){
    if (mode == "ox")
      indices <- which(df_copy[,cx]=="0")  
    else if (mode == "co2")
      indices <- which(df_copy[,cx]=="1")
    # look at the sum of the column to determine if there are more zeros or ones
    if (sum(as.integer(df_copy[,cx])) < length(df_copy[,cx])/2)
      df_copy <- df_copy[indices,]
    else
      df_copy <- df_copy[-indices,]
    if(length(df_copy) == ncol(df))
      break
  }
  return(df_copy)
}

get_crit2(tst_file, "ox")
get_crit2(tst_file, "co2")

get_life_supp <- function(vec){
  #browser()
  oxrate  <- as.integer(get_crit2(vec, "ox"))
  co2rate <- as.integer(get_crit2(vec, "co2"))
  bin_vec <- 2^((length(oxrate)-1):0)
  return(sum(oxrate * bin_vec)*sum(co2rate * bin_vec))
}
get_life_supp(tst_file)

get_life_supp(scan_zero)