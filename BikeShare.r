# load packages

library(tidyverse)
library(tidymodels)
library(vroom)
library(forcats)
library(patchwork)

# read in data

train <- vroom('train.csv')
test <- vroom('test.csv')

# ensure the weather variable is a factor and rename the values to something more interpretable

train <- train |>
  mutate(weather = factor(weather,
                          levels = c("1", "2", "3", "4"),
                          ordered = TRUE))
train <- train |>
  mutate(weather = fct_recode(weather,
                              "Clear"   = "1",
                              "Cloudy"  = "2",
                              "Light Precip"   = "3",
                              "Heavy Precip"  = "4"))

# ensure the season variable is a factor and rename the values to something more interpretable

train <- train |>
  mutate(season = factor(season,
                          levels = c("1", "2", "3", "4"),
                          ordered = TRUE))
train <- train |>
  mutate(season = fct_recode(season,
                              "Spring"   = "1",
                              "Summer"  = "2",
                              "Fall"   = "3",
                              "Winter"  = "4"))


# EXPLORATORY DATA ANALYSIS

glimpse(train)

DataExplorer::plot_intro(train)

DataExplorer::plot_histogram(train)

# plot the number of rentals during each weather category

weather <- ggplot(train, aes(x = fct_rev(weather))) +
  geom_bar(fill = 'steelblue') +
  geom_text(stat = "count", aes(label = ..count..), hjust = -0.2) +
  coord_flip()+
  labs(x = "Weather", y = "Count", title = "Number of Rentals During Each Weather Category") +
  theme_minimal()

# plot the number of rentals over time

weekly <- train |>
  mutate(week = floor_date(datetime, unit = "week")) |>
  group_by(week) |>
  summarise(value = sum(count), .groups = "drop")

time_series <- ggplot(weekly, aes(x = week, y = value)) +
  geom_line(color = "steelblue") +
  labs(x = "Week", y = "Count", title = "Rentals Over Time") +
  theme_minimal()

# plot correlation between each pair of variables

correlations <- DataExplorer::plot_correlation(train, geom_text_args = list(size = 0))

# plot casual vs. registered

casual_prop <- sum(train$casual)/sum(train$count)

registered_prop <- sum(train$registered)/sum(train$count)

df_props <- data.frame(
  Type = c("Casual", "Registered"),
  proportion = c(casual_prop, registered_prop)
)

type <- ggplot(df_props, aes(x = "Rentals", y = proportion, fill = Type)) +
  geom_col() +
  scale_fill_manual(values = c("Casual" = "steelblue", 
                               "Registered" = "darkgrey")) +
  labs(x = NULL, y = "Proportion",
       title = "Proportion of Casual vs Registered Rentals") +
  theme_minimal()

(weather + time_series) / (correlations + type)

# LINEAR REGRESSION - FIRST ATTEMPT

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>%
  fit(formula=count~.-datetime, data=train)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test) # Use fit to predict
bike_predictions ## Look at the output

kaggle_submission <- bike_predictions %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

# LINEAR REGRESSION - SECOND ATTEMPT

# cleaning

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))


# feature engineering

my_recipe <- recipe(count~., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_mutate(season = factor(season)) %>%
  step_corr(all_numeric_predictors(), threshold=0.8)

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = test)

# print baked dataset

print(head(baked, n = 5))

# workflow

linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

first_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(linear_model) %>%
  fit(data = train)

# make prediction

lin_preds <- predict(first_workflow, new_data = test)
lin_preds <- lin_preds %>% mutate(exp(.pred))


# prepare for kaggle submission

kaggle_submission_2 <- bike_predictions %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=kaggle_submission_2, file="./LinearPreds.csv", delim=",")

