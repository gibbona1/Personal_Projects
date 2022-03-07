#https://adventofcode.com/2021/day/8

getwd()

myfile <- readLines("input_day08.txt")

tmp_file <- 'be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce'

library(data.table)

write(tmp_file, "tmp_input_08.txt")
tmp_file <- readLines("tmp_input_08.txt")

#tmp_file
#myfile
get_num_segments <- function(filenm){
  tmp_list <- sapply(filenm, function(x) strsplit(x, " | ", fixed = TRUE))
  
  tmp_list[[1]][2]
  outputs <- as.vector(sapply(tmp_list, "[[", 2))
  
  output_df <- as.data.frame(sapply(outputs, function(x) strsplit(x, " ")))
  
  output_df <- transpose(output_df)
  
  output_len_df <- apply(output_df, 1, nchar)
  
  #1 uses 2 segments
  #4 uses 4 segments
  #7 uses 3 segments
  #8 uses 7 segments
  return(sum(output_len_df %in%  c(2, 3, 4, 7)))
}

get_num_segments(tmp_file)
get_num_segments(myfile)

get_sum_outputs <- function(filenm){
  tmp_list <- sapply(filenm, function(x) strsplit(x, " | ", fixed = TRUE))
  
  tmp_list[[1]][2]
  outputs <- as.vector(sapply(tmp_list, "[[", 2))
  
  output_df <- as.data.frame(sapply(outputs, function(x) strsplit(x, " ")))

  output_df <- transpose(output_df)
  
  output_len_df <- apply(output_df, 1, paste0)
  
  #1 uses 2 segments
  #4 uses 4 segments
  #7 uses 3 segments
  #8 uses 7 segments
  return(sum(output_len_df %in%  c(2, 3, 4, 7)))
}
tmp_list <- sapply(tmp_file, function(x) strsplit(x, " | ", fixed = TRUE))

outputs <- as.vector(sapply(tmp_list, "[[", 2))

output_df <- transpose(as.data.frame(sapply(outputs, function(x) strsplit(x, " "))))

strSort <- function(x) sapply(lapply(strsplit(x, NULL), sort), paste, collapse="")

for(i in 1:nrow(output_df)){
  for(j in 1:ncol(output_df)){
    output_df[i,j] = strSort(output_df[i,j])
  }
}

for(i in 1:nrow(output_df)){
  for(j in 1:ncol(output_df)){
    output_df[i,j] = strSort(output_df[i,j])
  }
}

output_df

sapply(c("abcdeg", "ab", "acdfg"), get_seg_val)

get_seg_val <- function(x) switch(x,
       "abcdeg"  = 0,
       "ab"      = 1,
       "acdfg"   = 2,
       "abcdf"   = 3,
       "abef"    = 4,
       "bcdef"   = 5,
       "bcdefg"  = 6,
       "abd"     = 7,
       "abcdefg" = 8,
       "abcdef"  = 9,
       NA)
       
myfile

output_num_df <- matrix(NA, nrow = nrow(output_df), ncol = ncol(output_df))
for(i in 1:nrow(output_num_df)){
  for(j in 1:ncol(output_num_df)){
    output_df[i,j] = get_seg_val(strSort(output_df[i,j]))
  }
}

output_num_df
