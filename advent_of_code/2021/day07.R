#https://adventofcode.com/2021/day/7

getwd()

myfile <- readLines("input_day07.txt")

my_pos <- as.integer(strsplit(myfile, ",")[[1]])

tmp_pos <- as.integer(strsplit('16,1,2,0,4,2,7,1,2,14', ',')[[1]])

get_min_cost <- function(vec){
  pos_range <- range(vec)
  pos_nums  <- pos_range[1]:pos_range[2]
  
  pos_nums
  costs <- sapply(pos_nums, function(x) sum(abs(vec-x)))
  return(min(costs))
}

get_min_cost(tmp_pos)
get_min_cost(my_pos)

get_min_cost2 <- function(vec){
  pos_range <- range(vec)
  pos_nums  <- pos_range[1]:pos_range[2]
  
  incr_dist <- function(x) return(x*(x+1)/2)
  costs <-sapply(pos_nums, function(x) sum(incr_dist(abs(vec-i))))
  return(min(costs))
}

get_min_cost2(tmp_pos)
get_min_cost2(my_pos)
