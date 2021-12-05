#Day 1
myfile <- read.table('input_2015_day01.txt')

bool_vec <- c()
for(tx in strsplit(as.character(myfile), split = "")){
  bool_vec <- c(bool_vec, tx == "(")
}

move_vec <- (bool_vec * 1) - (1-bool_vec * 1)
which(cumsum(move_vec) == -1)

#Day 2
myfile <- read.table('input_2015_day02.txt')
myfile

my_df <- data.frame(l = c(), w = c(), h = c())

for(i in 1:nrow(myfile)){
  str_vec <- strsplit(myfile[i,], "x")[[1]]
  #print(str_vec)
  new_df  <- data.frame(l = as.numeric(str_vec[1]),
                        w = as.numeric(str_vec[2]),
                        h = as.numeric(str_vec[3]))
  my_df   <- rbind(my_df, new_df)
}

which_max <- apply(my_df, 1, which.max)

my_df$maxind <- which_max

my_df$remainder_area <- apply(my_df, 1, function(x) prod(x[(1:3)[-x[4]]]))

my_df$main_area <- 2*(my_df[,1]*my_df[,2]+
                      my_df[,2]*my_df[,3]+
                      my_df[,1]*my_df[,3])

my_df$total_area <- my_df$main_area+my_df$remainder_area

sum(my_df$total_area)

#Part 2
my_df$bow <- apply(my_df, 1, function(x) prod(x[(1:3)]))

my_df$ribbon <- apply(my_df, 1, function(x) 2*sum(x[(1:3)[-x[4]]]))

my_df$total_ribbon <- my_df$ribbon + my_df$bow

sum(my_df$total_ribbon)

