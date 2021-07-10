# Verneek ML Interview project

The task here to build a recommendation model for research papers. The data we have is pulled from a website in which users can select the research papers that they like/are relevant to them. We only have have positive data, i.e. we know the papers a user likes, but don't have any data on user dislikes.

## Data
For each paper we have the following information:
* title
* abstract
* list of tags
* paper citations
* users that like them

Some of the articles have their likes removed from original dataset. This is simulate a situation in which new articles are added to the site. We want to be able to recommend these new articles to our current users. The test data is hidden from you.  The code you write needs to be able to generate user interest predictions for each article, and we'll use the code to evaluate on the test data.

You can use any subset of the data listed above as features. There is no requirement or expectation to use all of the types of features listed above. Trying to use all the features above may prevent you from finishing on-time.

## Code
Included here are some simple functions to help you along. 

In the data_loaders directory, there are functions to help you load the raw data into pandas or numpy data structures. You may run into trouble loading the data yourself, so using the provided data loaders will make your life easier. Feel free to convert the data once loaded into any format you want. 

The load_articles function will load article text into a pandas dataframe. In that dataframe you may find columns containing id numbers. These should be ignored. The ids for the articles are the row ids of the articles in the pandas dataframe. 

In the evaluation directory you can find the recall@M function which computes our evaluation metric. This is the metric that will be run on your final model output. Since you don't have access to test data, you won't be able to evaluate the final results, but you can use the function to do validation evaluations, etc.

## Input and Output

The input, after loading, is a matrix of size number of users X number of articles in the training set. There are 1s in locations where users like articles, 0s otherwise.

The expected output is a matrix of size number of users X number of articles in the test set. Each element in the matrix should contain a score estimating how well the user may like the article. Higher scores should indicate higher predicted interest. Outputing a score is important, since the evaluation will take into account ranking of scores and not just the predictions.

The test articles range from ids: 13584 - 16980 (the last 3396 articles in the dataset). So the test set matrix should have shape (5551, 3396).


## Requirements

1. Notebooks should only be used to do data analysis/visualization.
2. Model inference should be relatively fast/efficient.
3. Code should be clear and documented when needed.
4. It should be easy to start model training and inference from the command line.
5. You can use machine learning libraries such as scikit-learn, pytorch, pandas, pretrained huggingface transformers etc. but we are looking to see how you'd implement your own model on top of these tools. 
6. If you do copy any significant amount of code from internet even if you modify it, please attribute it with a comment with link and information about why you copied this code.
7. Commit your current state into a git repo regularly, at least every 20 mins. Part of the evaluation is seeing how your code evolves over the interview time.


You can use any IDE, and setup you'd like. You should create a private github repo and check-in this directory as well your code as you go along. 
 
Feel free to use any free resource for access to compute, such as google colab. After your final submission, we will train the final model for a couple of hours, and get the final evaluation. 

## Suggestions

You have a limited amount of time, so it makes sense to use the tools and libraries you are most familiar with so that you don't waste time learning new libraries. 

Keep things simple. You'll be asked to describe why you chose to use the libraries you did.

You can use any IDE and setup you'd like. You should create a private github repo and check-in the code as you go along.

Feel free to use any free resource for access to compute, such as google colab. After your final submission, we will train the final model for a couple of hours, and get the final evaluation.

You can code everything up in notebook to start and move things to library over time, but if that's the case try to code it up in a way that would make it easy to transfer.


Please document how to use your code in this file below. We will use the documentation listed here to run the final model training and evaluation.


# Candidate Documentation
Libraries required - 
* TensorFlow (1.7+) Please do not use TensorFlow 2
* Numpy, Pandas
* Pickle, ArgParse, OS


### Usage
The data folder contains the entire dataset required for training the model.
Run the files 'main.py' for training and testing of the implementation.

*Examples*

python main.py --train_data=data/train_data.dat --test_data=data/test_data.dat --test=True --n_users=5551 --a_train=13584 --a_test=1584

*Arguments*

train_data = path to training data

test_data = path to test data

n_users = # Users


a_train = Number of articles in train set

a_test = number of articles in test set

Hyper-parameter tuning and exploration has not been possible with the time constrain. Hence, the following parameters has been chosen.

Dropout ratio of 0.1 has been selected.

LR = 0.03

Batch-size = 256

Epochs =50-150

Learning rate has 
### Idea Formulation
The idea is based on the research paper **Collaborative Deep Learning for Recommender Systems** by Hao Wang. 
It introduced the Collaborative Deep Learning model based on Hierarchical Bayesian architecture for Recommendation Systems.
The main deifference between the implementation of the original paper and this model architecture is that the SDAE encoder-decoder structure is not used here.
A simple 2 layer encoder-decoder is used in this implementation for the purpose of simplicity and time constraints.

For the purpose of time constraint, I am only taking into account the training data in 'train.dat' and then preparing the article to content matching using
(article_id -> bag of words of content), the bag f contents representation provided in the dataset.

**Data Dimesnsions**

Train Data -> #Users X #Training_articles

Item Information Data -> #Total_Articles X Vocab_dimension

The evaluation has been performed using the function already rovided to measure the mAP scores. Referring to the research paper, the current implementation achieves comparable mAP50 scores as denoted in the graphs of the paper.


# REFERENCES

* The idea has been taken from the research paper Collaborative Deep Learning for Recommender Systems by Hao Wang.

The code sources has been taken from the official github repo of the research paper.
* https://github.com/js05212/CollaborativeDeepLearning-TensorFlow
* https://github.com/gtshs2/Collaborative_Deep_Learning
* https://github.com/zoujun123/Keras-CDL
* https://github.com/xiaoouzhang/Collaborative-Deep-Learning-for-Recommender-Systems