
## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula=count~X1+X2+..., data=train)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test) # Use fit to predict
bike_predictions ## Look at the output