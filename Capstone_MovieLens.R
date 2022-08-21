
# This code scripts is about the MovieLens project in the course HarvardX: ph125.9x Data Science: Capstone Project
# Author: Juan Du

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(lubridate)
library(dslabs)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Data Exploration and Visualization
##########################################################

# edx dataset and its features
head(edx)

# number of rows and columns are there in the edx dataset
dim(edx)

# number of rows and columns are there in the edx dataset
dim(validation)

# number of unique users that provided ratings and number of unique movies were rated in edx dataset
edx %>% summarise(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId))

# Average rating in edx dataset
mean(edx$rating)

# Rating distribution
rating_distribution <- as.vector(edx$rating)
unique(rating_distribution)

# Visualize rating distribution
rating_distribution <- rating_distribution[rating_distribution !=0]
rating_distribution <- factor(rating_distribution)
qplot(rating_distribution) + ggtitle("Rating Distribution ")


# User distribution
edx %>% count(userId) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 50, color="blue") +
  scale_x_log10() +
  ggtitle("User Distribution")

# Movie Distribution
edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color="blue") +
  scale_x_log10()+
  ggtitle("Movie Distribution")




#################################################################
# Prediction models 
#################################################################


# Build a function that computes the RMSE for vectors of ratings and their predictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Model 1: Just Mean average model
mu_hat <- mean(edx$rating)
mu_hat

# Predict all unknown ratings with mu_hat and calculate RMSE
mu_rmse <- RMSE(validation$rating, mu_hat)
mu_rmse

# Save the Model 1 results
rmse_results <- data_frame(method = "Just the mean average", 
                                     RMSE = mu_rmse)
rmse_results %>% knitr::kable()

# Model 2: Movie-effect Model
mu <- mean(edx$rating)
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

# Plot movie rating variability
qplot(b_i, data = movie_avgs, bin = 30, color = "blue")

# Predict movie-effect model and calculate RMSE
predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)
bi_rmse <- RMSE(predicted_ratings, validation$rating)
bi_rmse

# Save the Model 2 result
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie-effect model", RMSE =bi_rmse))
rmse_results %>% knitr::kable()


# Model 3: Movie and User-effect Model
# Compute mu and b_i and estimate b_u
user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu -b_i))

#Construct predictors 
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#calculate RMSE
bu_rmse <- RMSE(predicted_ratings, validation$rating)
bu_rmse

# Save the model 3 RMSE results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Movie and user effect model",
                                     RMSE = bu_rmse))
rmse_results %>% knitr::kable()

# Model 4: Regularized Movie-Effect Model

# Calculate optimal lambda
lambdas <- seq(0, 10, 0.25)

mu <- mean(edx$rating)

just_the_sum <- edx %>%
  group_by(movieId) %>%
  summarise(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>%
    left_join(just_the_sum, by = 'movieId') %>%
    mutate(b_i = s/(n_i +l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
return(RMSE(predicted_ratings, validation$rating))
  })

qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda

# Calculate regularized movie-effect RMSE
movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/ (n() + lambda), n_i=n())

predicted_ratings <- validation %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# calculate Model 4 RMSE
movie_reg_rmse <- RMSE(predicted_ratings, validation$rating)

# Save the Model 4 RMSE result
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized movie-effect model",
                                     RMSE = movie_reg_rmse))
rmse_results %>% knitr::kable()

# Model 5: Regularized movie-effect and user-effect model

# Calculate optimal lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/ (n() + l))

  b_u <- edx %>%
    left_join(b_i, by = 'movieId') %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu) / (n() +l ))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
  })

qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

# Calculate regularized movie and user effect RMSE

movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu) /(n() + lambda), n_i = n())

user_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/(n() +lambda), n_u = n())

predicted_ratings <- validation %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate model 5 RMSE
movie_user_reg_rmse <- RMSE(predicted_ratings, validation$rating)

# Save the result
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized movie and user effect model",
                                     RMSE = movie_user_reg_rmse))
rmse_results %>% knitr::kable()

