# Getting started with word_embedding in R
# http://programminghistorian.github.io/ph-submissions/lessons/getting-started-with-word-embeddings-in-r#fnref:4


setwd("/Users/Allen/Documents/ISM") 
library(devtools)
library(wordVectors)
library(tsne)
library(Rtsne)
library(ggplot2)
library(ggrepel)

#combines the texts into one document, removes punctuation, and converts all words to lower case.
prep_word2vec("Preys/text.txt", "Pepys_processed.txt", lowercase = T)
# To prepare a folder of plain text files:
# prep_word2vec("PATH-TO-FOLDER-WHERE-FILES-ARE", "NAME-OF-CREATED-CORPUS-FILE.txt", lowercase = T)

# Training the Model （word2vec）
# use default skip-gram method of creating the word embedding, which is better for infrequent words
pepys <- train_word2vec("Pepys_processed.txt",
                        output = "Pepys_model.bin", threads = 1,
                        vectors = 100, window = 12)

## Exploring the Word Embedding Model
#1 Closest Words to a Chosen Term
closest_to(pepys, "england", n = 10)

# write a function w2v_plot that will create a subset of the whole model which focuses on a chosen term
w2v_plot <- function(model, word, path, ref_name) {
  
  # Identify the nearest 10 words to the average vector of search terms
  ten <- nearest_to(model, model[[word]])
  
  # Identify the nearest 500 words to the average vector of search terms and
  # save as a .txt file
  main <- nearest_to(model, model[[word]], 500)
  wordlist <- names(main)
  filepath <- paste0(path, ref_name)
  write(wordlist, paste0(filepath, ".txt"))
  
  # Create a subset vector space model
  new_model <- model[[wordlist, average = F]]
  
  # Run Rtsne to reduce new Word Embedding Model to 2D (Barnes-Hut)
  reduction <- Rtsne(as.matrix(new_model), dims = 2, initial_dims = 50,
                     perplexity = 30, theta = 0.5, check_duplicates = F,
                     pca = F, max_iter = 1000, verbose = F,
                     is_distance = F, Y_init = NULL)
  
  # Extract Y (positions for plot) as a dataframe and add row names
  df <- as.data.frame(reduction$Y)
  rows <- rownames(new_model)
  rownames(df) <- rows
  
  # Create t-SNE plot and save as jpeg
  # create a folder in your working directory called Results
  
  
  ggplot(df) +
    geom_point(aes(x = V1, y = V2), color = "red") +
    geom_text_repel(aes(x = V1, y = V2, label = rownames(df))) +
    xlab("Dimension 1") +
    ylab("Dimension 2 ") +
    # geom_text(fontface = 2, alpha = .8) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none") +
    ggtitle(paste0("2D reduction of Word Embedding Model ", ref_name," using t_SNE"))
  
  ggsave(paste0(ref_name, ".jpeg"), path = path, width = 24,
         height = 18, dpi = 100)
  
  new_list <- list("Ten nearest" = ten, "Status" = "Analysis Complete")
  return(new_list)
}

w2v_plot(pepys, "king", "result/", "king") #output plot and text 


###Clustering 
#kmeans algorithm return the top ten terms for ten topics
set.seed(40)
centers = 100
clustering = kmeans(pepys,centers=centers,iter.max = 40)

sapply(sample(1:centers,10),function(n) {
  names(clustering$cluster[clustering$cluster==n][1:10])
})

# Examining Terms of Interest in Context ： identify sections of text which may be of particular interest.
# Create a path to the files
input.dir <- "Pepys"
# Read the name of all .txt files
files <- dir(input.dir, "\\.txt")

make_word_list <- function(files, input.dir) {
  # create an empty list for the results
  word_list <- list()
  # read in the files and process them
  for(i in 1:length(files)) {
    text <- scan(paste(input.dir, files[i], sep = "/"),
                 what = "character", sep = "\n")   
    text <- paste(text, collapse = " ")
    text_lower <- tolower(text)
    text_words <- strsplit(text_lower, "\\W")
    text_words <- unlist(text_words)
    text_words <- text_words[which(text_words != "")]
    word_list[[files[i]]] <- text_words
  }
  return(word_list)
}

#kwic function
kwic <- function(files, input, word, context) {
  corpus <- make_word_list(files, input)
  context <- as.numeric(context)
  keyword <- tolower(word)
  result <- NULL
  # create the KWIC readout
  for (i in 1:length(corpus)) {
    hits <- which(corpus[[i]] == keyword)
    doc <- files[i]
    if(length(hits) > 0){
      for(j in 1:length(hits)) {
        start <- hits[j] - context
        if(start < 1) {
          start <- 1
        }
        end <- hits[j] + context
        myrow <- cbind(doc, hits[j],
                       paste(corpus[[i]][start: (hits[j] -1)],
                             collapse = " "),
                       paste(corpus[[i]][hits[j]],
                             collapse = " "),
                       paste(corpus[[i]][(hits[j] +1): end],
                             collapse = " "))
        result <- rbind(result, myrow)
      }
      
    } else {
      z <- paste0(doc, " YOUR KEYWORD WAS NOT FOUND\n")
      cat(z)
    }
  }
  colnames(result) <- c("file", "position", "left",
                        "keyword", "right")
  write.csv(result, paste0("Results/", word, "_",
                           context, ".csv"))
  cat("Your results have been saved")
}

kwic(files, input.dir, "pardoned", 6)

# The results can be read back into R if desired:
results <- read.csv("Results/pardoned_6.csv", header = T)

# Remove column added during the conversion
results <- results[, -1]

# View the results
View(results)
