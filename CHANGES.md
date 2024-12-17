## Changes
This file is to denote the chanes done by Cagan Ahmet Yanmaz to the project. There'll be some significant changes

Slight bug fix while calculating mean success rates of companies (the current method is incorrect due to joint investments etc)

## 1. Dataset Prep
Completely ditched the long short csv files, as coinvestment dataset provide better information. Also split the companies as training, test (from a specific year)


## 2. Features

In order to calculate the effectiveness of our model, we need to have a test data as well. Also, features such as "success buckets" require as to know the success rate beforehand, which is our objective (determining the probability of the success of a given company with a given investor), so we can't just split the dataset into training and testing. Instead, I decided to use companies before 2022 to develop the model, and use the companies after 2022 as test. Note that this is still not perfect, as successful compaines before 2022 might've got successful after 2022, which we wouldn't have known if we used the model in 2022. So this is mostly theoretical and imperfect.

A further work might be to fix that.



