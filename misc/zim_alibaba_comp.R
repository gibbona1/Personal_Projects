#install.packages("quantmod")

library(quantmod)
library(ggplot2)
library(dplyr)
library(plotly)

# Define the symbols
symbols <- c("BABA", "ZIM")

# Download the stock prices
getSymbols(symbols, from = start_date, to = end_date)

# Access the stock price data
baba_prices <- BABA[, "BABA.Close"]
zim_prices  <- ZIM[, "ZIM.Close"]

frac_price <- data.frame(Date = index(baba_prices),
                         #val  = as.vector(baba_prices/zim_prices))
                         val = as.vector(1/((1500-215)/82/baba_prices) - zim_prices))

frac_price %>% filter(Date >= "2022-08-01") %>%
  #ggplot() + geom_line(aes(x=Date, y=val)) 
  plot_ly(x = ~Date, y = ~val, type = "scatter", mode = "lines")


150/13.27
1/((1700-215)/82/175) - 10.26

10.26*200
1700*80
