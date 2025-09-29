# load packages

library(tidyverse)
library(tidymodels)
library(vroom)
library(forcats)
library(patchwork)
library(bonsai)
library(lightgbm)
library(agua)


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
lin_preds <- lin_preds %>% mutate(.pred = exp(.pred))


# prepare for kaggle submission

kaggle_submission_2 <- lin_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=kaggle_submission_2, file="./LinearPreds2.csv", delim=",")

# PENALIZED REGRESSION - FIRST 5 ATTEMPTS

# cleaning

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))


# feature engineering

preg_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)


# workflow 1

preg_model <- linear_reg(penalty = 0, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

preg_workflow <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(preg_model) %>%
  fit(data = train)

# make prediction 1

preg_0_1_preds <- predict(preg_workflow, new_data = test)
preg_0_1_preds <- preg_0_1_preds %>% mutate(.pred = exp(.pred))


# prepare for kaggle submission 1

preg_sub_0_1 <- preg_0_1_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = preg_sub_0_1,
  file = "./PregPreds0_1.csv",
  delim = ","
)

# workflow 2

preg_model2 <- linear_reg(penalty = 0.001, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

preg_workflow2 <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(preg_model2) %>%
  fit(data = train)

# make prediction 2

preg_001_1_preds <- predict(preg_workflow2, new_data = test)
preg_001_1_preds <- preg_001_1_preds %>% mutate(.pred = exp(.pred))


# prepare for kaggle submission 2

preg_sub_001_1 <- preg_001_1_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = preg_sub_001_1,
  file = "./PregPreds001_1.csv",
  delim = ","
)

# workflow 3

preg_model3 <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

preg_workflow3 <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(preg_model3) %>%
  fit(data = train)

# make prediction 3

preg_point1_0_preds <- predict(preg_workflow3, new_data = test)
preg_point1_0_preds <- preg_point1_0_preds %>% mutate(.pred = exp(.pred))


# prepare for kaggle submission 3

preg_sub_point1_0 <- preg_point1_0_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = preg_sub_point1_0,
  file = "./PregPredspoint1_0.csv",
  delim = ","
)


# workflow 4

preg_model4 <- linear_reg(penalty = 5, mixture = .5) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

preg_workflow4 <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(preg_model4) %>%
  fit(data = train)

# make prediction 4

preg_5_point5_preds <- predict(preg_workflow4, new_data = test)
preg_5_point5_preds <- preg_5_point5_preds %>% mutate(.pred = exp(.pred))


# prepare for kaggle submission 4

preg_sub_5_point5 <- preg_5_point5_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = preg_sub_5_point5,
  file = "./PregPreds5_point5.csv",
  delim = ","
)

# workflow 5

preg_model5 <- linear_reg(penalty = 1, mixture = .2) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

preg_workflow5 <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(preg_model5) %>%
  fit(data = train)

# make prediction 5

preg_1_point2_preds <- predict(preg_workflow5, new_data = test)
preg_1_point2_preds <- preg_1_point2_preds %>% mutate(.pred = exp(.pred))


# prepare for kaggle submission 5

preg_sub_1_point2 <- preg_1_point2_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = preg_sub_1_point2,
  file = "./PregPreds1_point2.csv",
  delim = ","
)

# PENALIZED REGRESSION PARAMETER TUNING

# cleaning

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))


# feature engineering

preg_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)

# model

tuned_preg_model <- linear_reg(penalty = tune(),
                               mixture = tune()) %>%
  set_engine("glmnet")

# workflow

tuned_preg_wf <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(tuned_preg_model)

# tuning grid

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)

# split for CV

folds <- vfold_cv(train, v = 5, repeats = 1)

# run the cv

cv_results <- tuned_preg_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse))

# plot results

collect_metrics(cv_results) %>%
  ggplot(data = ., aes(x=penalty, y = mean, color = factor(mixture))) +
  geom_line()

# find best parameters

best_tune <- cv_results %>%
  select_best(metric = "rmse")

# finalize and fit wf

final_wf <-
  tuned_preg_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data= train)

# predict

tuned_preds <- final_wf %>%
  predict(new_data =  test)
tuned_preds <- tuned_preds %>% mutate(.pred = exp(.pred))


# prepare for tuned kaggle submission

tuned_preds_sub <- tuned_preds %>%
  bind_cols(., test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = tuned_preds_sub,
  file = "./PregPreds.csv",
  delim = ","
)

# DECISION TREE

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))

# feature engineering
tree_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)

# model
tree_model <- decision_tree(tree_depth = tune(),
                            cost_complexity = tune(),
                            min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# workflow
tree_wf <- workflow() %>%
  add_recipe(tree_recipe) %>%
  add_model(tree_model)

# tuning grid
tree_grid <- grid_regular(tree_depth(),
                          cost_complexity(),
                          min_n(),
                          levels = 5)

# split for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

# run the cv
tree_cv_results <- tree_wf %>%
  tune_grid(resamples = folds,
            grid = tree_grid,
            metrics = metric_set(rmse))

# find best parameters
best_tree_tune <- tree_cv_results %>%
  select_best(metric = "rmse")

# finalize and fit workflow
final_tree_wf <- tree_wf %>%
  finalize_workflow(best_tree_tune) %>%
  fit(data = train)

# predict
tuned_tree_preds <- final_tree_wf %>%
  predict(new_data = test) %>%
  mutate(.pred = exp(.pred))

# prepare for kaggle submission
tree_preds_sub <- tuned_tree_preds %>%
  bind_cols(test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = tree_preds_sub,
  file = "./DecisionTreePreds.csv",
  delim = ","
)

# RANDOM FOREST


train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))

# feature engineering
forest_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)

# model
forest_model <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# workflow
forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(forest_model)

# tuning grid
forest_grid <- grid_regular(mtry(range = c(1,8)),
                            min_n(),
                            levels = 5)

# split for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

# run the cv
forest_cv_results <- forest_wf %>%
  tune_grid(resamples = folds,
            grid = forest_grid,
            metrics = metric_set(rmse))

# find best parameters
best_forest_tune <- forest_cv_results %>%
  select_best(metric = "rmse")

# finalize and fit workflow
final_forest_wf <- forest_wf %>%
  finalize_workflow(best_forest_tune) %>%
  fit(data = train)

# predict
tuned_forest_preds <- final_forest_wf %>%
  predict(new_data = test) %>%
  mutate(.pred = exp(.pred))

# prepare for kaggle submission
forest_preds_sub <- tuned_forest_preds %>%
  bind_cols(test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = forest_preds_sub,
  file = "./RandomForestPreds.csv",
  delim = ","
)

# BOOSTED MODEL (BART)

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))

# feature engineering
bart_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)

# model
bart_model <- bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

# workflow
bart_wf <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_model)

# tuning grid
bart_grid <- grid_regular(trees(),
                          levels = 5)

# split for CV
folds <- vfold_cv(train, v = 5, repeats = 1)

# run the cv
bart_cv_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_grid,
            metrics = metric_set(rmse))

# find best parameters
best_bart_tune <- bart_cv_results %>%
  select_best(metric = "rmse")

# finalize and fit workflow
final_bart_wf <- bart_wf %>%
  finalize_workflow(best_bart_tune) %>%
  fit(data = train)

# predict
tuned_bart_preds <- final_bart_wf %>%
  predict(new_data = test) %>%
  mutate(.pred = exp(.pred))

# prepare for kaggle submission
bart_preds_sub <- tuned_bart_preds %>%
  bind_cols(test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = bart_preds_sub,
  file = "./BARTPreds.csv",
  delim = ","
)

# STACKED MODEL

h2o::h2o.init()

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))

# feature engineering
stacked_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)


stacked_model <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs = 300) %>%
  set_mode("regression")

stacked_model_wf <- workflow() %>%
  add_recipe(stacked_recipe) %>%
  add_model(stacked_model) %>%
  fit(data = train)

h2o_preds <- stacked_model_wf %>%
  predict(new_data = test) %>%
  mutate(.pred = exp(.pred))

# prepare for kaggle submission
h2o_preds_sub <- h2o_preds %>%
  bind_cols(test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = h2o_preds_sub,
  file = "./h2oPreds.csv",
  delim = ","
)
      
      
## DataRobot Dataset

train <- vroom('train.csv')
test <- vroom('test.csv')

train <- train |> select(1:9, 12)
train <- train |> mutate(count = log(count))

# define recipe
robot_recipe <- recipe(count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.8)

# prep the recipe on the training data
robot_prep <- prep(robot_recipe, training = train)

# bake the recipe on TRAINING data
baked_train <- bake(robot_prep, new_data = train)

# bake the recipe on TEST data
baked_test <- bake(robot_prep, new_data = test)

readr::write_csv(baked_train, "baked_train.csv")
readr::write_csv(baked_test, "baked_test.csv")

robot_preds <- vroom('datarobotpreds.csv') %>%
  mutate(count_PREDICTION = exp(count_PREDICTION))

# prepare for kaggle submission
robot_preds_sub <- robot_preds %>%
  bind_cols(test) %>%
  select(datetime, count_PREDICTION) %>%
  rename(count = count_PREDICTION) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(
  x = robot_preds_sub,
  file = "./RobotPreds.csv",
  delim = ","
)
