#https://adventofcode.com/2021/day/1

getwd()
setwd('C:/Users/Anthony/Downloads/AdventOfCode')
myfile <- read.table('input_day01.txt')

#Part 1
count <- 0
for(i in 1:nrow(myfile)){
  if(i == 1)
    label <- "N/A - no previous measurement"
  else {
    if(myfile[i,] > myfile[i-1,]){
      count <- count + 1
      label <- "increased"
    } else
      label <- "decreased"
  }
  cat(i, ": ", paste0(myfile[i,], "\t(", label, ")\n"))
}
print(count)

#Part 2
count <- 0
sum3  <- c()
for(i in 1:(nrow(myfile)-2)){
  sum3 <- c(sum3, sum(myfile[i:(i+2),]))
}
sum3
for(i in 1:length(sum3)){
  if(i == 1)
    label <- "N/A - no previous measurement"
  else {
    if(sum3[i] > sum3[i-1]){
      count <- count + 1
      label <- "increased"
    } else if(sum3[i] < sum3[i-1]){
      label <- "decreased"
    } else {
        label <- "no change"
      }
  }
  cat(i, ": ", paste0(sum3[i], "\t(", label, ")\n"))
}
print(count)