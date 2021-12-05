#https://adventofcode.com/2021/day/4
getwd()
setwd('C:/Users/Anthony/Downloads/AdventOfCode')

input_lines <- readLines("input_day04.txt")

input_lines_clean <- input_lines[input_lines != ""]

num_list <- sapply(strsplit(input_lines_clean[1], ","), as.integer)[,1]

#bingo_squares <- 
square_lines <- input_lines_clean[-1]

square_df <- sapply(square_lines,function(x) strsplit(x, " "))

lapply(square_df, function(x) x[x != ""])

square_nums <- as.integer(unlist(lapply(square_df, function(x) x[x != ""])))

square_mat <- matrix(square_nums, ncol = 5, byrow = TRUE)

square_mat

bingo_squares <- lapply(1:length(num_list), 
                        function(ix) 
                          square_mat[(1:5)+5*(ix-1),]
                        )

bingo_squares

test_num_list <- c(7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1)

test_bingo_squares <- list(
  matrix(c(22, 13, 17, 11,  0,
            8,  2, 23,  4, 24,
           21,  9, 14, 16,  7,
            6, 10,  3, 18,  5,
            1, 12, 20, 15, 19), ncol = 5, byrow = TRUE),
  matrix(c( 3, 15,  0,  2, 22,
            9, 18, 13, 17,  5,
           19,  8,  7, 25, 23,
           20, 11, 10, 24,  4,
           14, 21, 16, 12,  6), ncol = 5, byrow = TRUE),
  matrix(c(14, 21, 17, 24,  4,
           10, 16, 15,  9, 19,
           18,  8, 23, 26, 20,
           22, 11, 13,  6,  5,
            2,  0, 12,  3,  7), ncol = 5, byrow = TRUE)
)

#win_sq <- c()

#for(num in test_num_list){
#  for(mx in 1:length(test_bingo_squares)){
#    mat <- test_bingo_squares[[mx]]
#    mat[mat == num] <- -1
#    if(any(colSums(mat) == 0) | any(rowSums(mat)==0))
#      win_sq <- c(win_sq, mx)
#  }
#}
#win_sq

find_win_square <- function(num_list, bin_squares){
  winning_card <- NULL
  for(current_number in num_list) {
    bin_squares <- lapply(bin_squares, function(mat) {
      mat[which(mat==current_number)] <- -1
      return(mat)
    })
    #print(bin_squares)
    for (card in bin_squares) {
      if (any(rowSums(card)==-5) | any(colSums(card)==-5)){
        winning_card <- card
        return(sum(winning_card[winning_card != -1])*current_number)
      }
    }
  }
}

find_win_square(test_num_list, test_bingo_squares)
find_win_square(num_list, bingo_squares)

find_win_square <- function(num_list, bin_squares){
  winning_scores <- list()
  winning_card_x <- rep(0, length(bin_squares))
  for(current_number in num_list) {
    bin_squares <- lapply(bin_squares, function(mat) {
      mat[which(mat==current_number)] <- -1
      return(mat)
    })
    #print(bin_squares)
    bin_squares_copy <- bin_squares
    if(length(bin_squares) == 0)
      return(winning_scores)
    for (cx in which(winning_card_x==0)) {
      card <- bin_squares_copy[[cx]]
      if (any(rowSums(card)==-5) | any(colSums(card)==-5)){
        #print(paste0("card:", cx))
        #print(paste0("current_number:", current_number))
        #print(paste0("card score:", sum(card[card != -1])))
        winning_scores <- append(winning_scores, sum(card[card != -1])*current_number)
        winning_card_x[cx] <- 1
      }
    }
  }
  #get last score
  return(winning_scores[[length(winning_scores)]])
}

find_win_square(test_num_list, test_bingo_squares)
find_win_square(num_list, bingo_squares)
