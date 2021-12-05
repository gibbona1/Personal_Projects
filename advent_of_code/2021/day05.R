#https://adventofcode.com/2021/day/5
getwd()
setwd('C:/Users/Anthony/Downloads/AdventOfCode')

tst_input <- c("0,9 -> 5,9",
               "8,0 -> 0,8",
               "9,4 -> 3,4",
               "2,2 -> 2,1",
               "7,0 -> 7,4",
               "6,4 -> 2,0",
               "0,9 -> 2,9",
               "3,4 -> 1,4",
               "0,0 -> 8,8",
               "5,5 -> 8,2")

myfile <- readLines("input_day05.txt")

get_num_ovlp <- function(input_file, just_hv = TRUE){
  points_lst <- sapply(input_file, function(x) strsplit(x, " -> "))
  
  points_df <- data.frame(x1=c(), y1=c(), x2=c(), y2=c())
  for(pnts in points_lst){
    p1 <- as.integer(strsplit(pnts[1], ",")[[1]])
    p2 <- as.integer(strsplit(pnts[2], ",")[[1]])
    points_df <- rbind(points_df, 
                       data.frame(x1=p1[1], y1=p1[2], 
                                  x2=p2[1], y2=p2[2]))
  }
  
  max_x <- max(c(points_df$x1, points_df$x2))
  max_y <- max(c(points_df$y1, points_df$y2))
  
  mat_size  <- (max_x+1)*(max_y+1)
  
  lines_mat <- matrix(rep(0, mat_size), ncol = max_x+1, byrow = TRUE)
  
  lines_df  <- points_df
  
  if(just_hv){
    hv_lines  <- points_df$x1 == points_df$x2 | points_df$y1 == points_df$y2
    lines_df  <- lines_df[hv_lines,]
  }
  
  for(i in 1:nrow(lines_df)){
    line_df <- lines_df[i,]
    if(line_df$x1 == line_df$x2){
      ys <- line_df$y1:line_df$y2
      xs <- rep(line_df$x1, length(ys))
    } else if(line_df$y1 == line_df$y2){
      xs <- line_df$x1:line_df$x2
      ys <- rep(line_df$y1, length(xs))
    } else{
      xs <- line_df$x1:line_df$x2
      ys <- line_df$y1:line_df$y2
    }
    for(j in 1:length(xs))
      lines_mat[xs[j]+1, ys[j]+1] <- lines_mat[xs[j]+1, ys[j]+1] + 1 
  }
  return(sum(lines_mat>=2))
}

get_num_ovlp(tst_input)
get_num_ovlp(tst_input, just_hv = FALSE)

get_num_ovlp(myfile)
get_num_ovlp(myfile, just_hv = FALSE)
