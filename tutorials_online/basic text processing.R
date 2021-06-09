# Basic Text Processing in R

# https://programminghistorian.org/lessons/basic-text-processing-in-r

library(tidyverse)
library(tokenizers)

# small example
text <- paste("Now, I understand that because it's an election season",
              "expectations for what we will achieve this year are low.",
              "But, Mister Speaker, I appreciate the constructive approach",
              "that you and other leaders took at the end of last year",
              "to pass a budget and make tax cuts permanent for working",
              "families. So I hope we can work together this year on some",
              "bipartisan priorities like criminal justice reform and",
              "helping people who are battling prescription drug abuse",
              "and heroin abuse. So, who knows, we might surprise the",
              "cynics again")
words <- tokenize_words(text)
length(words) #1 cuz function returns a list
length(words[[1]])
tab <- table(words[[1]])
tab <- data_frame(word = names(tab), count = as.numeric(tab))
tab
arrange(tab, desc(count))  #descending order

##
#Detecting Sentence Boundaries
sentences <- tokenize_sentences(text)
sentences
length(sentences)
sentence_words <- tokenize_words(sentences[[1]])   #separate words in each sentences
sentence_words
length(sentence_words)
sapply(sentence_words, length)  #number of words in each sentence

##
#Analyzing Barack Obama’s 2016 State of the Union Address
#Exploratory Analysis
#import data
base_url <- "https://programminghistorian.org/assets/basic-text-processing-in-r"
url <- sprintf("%s/sotu_text/236.txt", base_url)
text <- paste(readLines(url), collapse = "\n")

words <- tokenize_words(text)
length(words[[1]])
tab <- table(words[[1]])
tab <- data_frame(word = names(tab), count = as.numeric(tab))
tab <- arrange(tab, desc(count))
tab

# a dataset from Peter Norvig using the Google Web Trillion Word Corpus, 
# collected from data gathered via Google’s crawling of known English websites:

wf <- read_csv(sprintf("%s/%s", base_url, "word_frequency.csv"))
wf
tab <- inner_join(tab, wf) #combine two dataset
tab
filter(tab, frequency < 0.1)  #filter those larger frequency words 
print(filter(tab, frequency < 0.002), n = 15)


# Document Summarization
metadata <- read_csv(sprintf("%s/%s", base_url, "metadata.csv")) # supply contextual information
metadata
tab <- filter(tab, frequency < 0.002)
result <- c(metadata$president[236], metadata$year[236], tab$word[1:5])
paste(result, collapse = "; ")

###
## Analyzing Every State of the Union Address from 1790 to 2016
## Loading the Corpus
files <- sprintf("%s/sotu_text/%03d.txt", base_url, 1:236)
text <- c()
for (f in files) {
  text <- c(text, paste(readLines(f), collapse = "\n"))
}

# Exploratory Analysis
words <- tokenize_words(text)
sapply(words, length)

qplot(metadata$year, sapply(words, length)) #word count by year

qplot(metadata$year, sapply(words, length),   #written or orally
      color = metadata$sotu_type)

# Stylometric Analysis 
sentences <- tokenize_sentences(text)
sentence_words <- sapply(sentences, tokenize_words)

sentence_length <- list()
for (i in 1:nrow(metadata)) {
  sentence_length[[i]] <- sapply(sentence_words[[i]], length)
}

sentence_length_median <- sapply(sentence_length, median)
qplot(metadata$year, sentence_length_median) +
  geom_smooth()  #result implies shorter sentences

#Document Summarization
description <- c()
for (i in 1:length(words)) {
  tab <- table(words[[i]])
  tab <- data_frame(word = names(tab), count = as.numeric(tab))
  tab <- arrange(tab, desc(count))
  tab <- inner_join(tab, wf)
  tab <- filter(tab, frequency < 0.002)
  
  result <- c(metadata$president[i], metadata$year[i], tab$word[1:5])
  description <- c(description, paste(result, collapse = "; "))
}
cat(description, sep = "\n")
