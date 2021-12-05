#https://adventofcode.com/2021/day/2

getwd()
setwd('C:/Users/Anthony/Downloads/AdventOfCode')
myfile <- read.table('input_day02.txt')

myfile$id <- as.integer(row.names(myfile))
myfile_wide <- reshape(myfile, idvar = "id", timevar = "V1", direction = "wide")
myfile_wide[is.na(myfile_wide)] <- 0

myfile_wide


myfile_sum <- colSums(myfile_wide)

myfile_sum[2]*(myfile_sum[3]-myfile_sum[4])


#Part 2

myfile
aim <- depth <- horizontal <- 0
for(i in 1:nrow(myfile)){
  if(myfile$V1[i] == "down"){
    aim <- aim + myfile$V2[i]
  } else if(myfile$V1[i] == "up"){
    aim <- aim - myfile$V2[i]
  } else { #forward
    horizontal <- horizontal + myfile$V2[i]
    depth <- depth + aim*myfile$V2[i]
  }
}
horizontal*depth