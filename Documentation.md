# Documentation


## Introduction

We - Micaela Barkman, Kai Dönnebrink, and Tom Hatton - as participants of the block seminar "Machine Learning in Practice" taught by Lucas Bechberger in the summer term 2021 at the Universität Osnabrück implemented a simple machine learning pipeline.
It includes steps for preprocessing, feature extraction, classification, and evaluation.

The goal of our machine learning pipeline was to create a Machine Learning model that can predict whether a tweet goes viral.
In addition, the main focus was on the process of creating such a Machine Learning model.
We worked in an agile way and followed different development practices like TDD, pair programming, and clean code guidelines.

### Task Definition

The task for the Machine Learning model is to predict whether a tweet related to the domain of data science goes viral, i.e. receives a certain number of likes and retweets.
A tweet is defined as viral if the sum of likes and retweets is higher than a certain threshold.
Initially the threshold is set to `50` but it can be changed by the user.

### Data

We used the dataset **Data Science Tweets 2010-2021** which can be found on [kaggle](https://www.kaggle.com/ruchi798/data-science-tweets) as training data.
It contains tweets and some related meta information from verified accounts on Twitter that contain `data science`, `data analysis`, or `data visualization` from 2010 to 2021.
For all tweets 38 values are captured including, for example, the date and time the tweet was posted, the twitter user, the tweet itself, and many more.

---

## Evaluation

### Evaluation metrics

Evaluation metrics are used to make an informed decision about the quality of a classifier. 
We implemented seven metrics namely Accuracy, Precision, Recall, F1 score, Cohen's Kappa, Log Loss, and ROC AUC.

- **Accuracy:** Basic metrics that is not very reliable when dealing with an imbalanced data set. So, we just used it to evaluate the baselines performances but do not take it into account to choose the best performing Machine Learning model.
- **Precision:** It is the ability of the classifier to not label a negative sample as positive. Because we also implemented the F1 score we again did not use it to choose the best performing model.
- **Recall:** It is the ability of the classifier to label all positive samples correctly. We also did not use it directly because we also implemented the F1 score.
- **F1 score:** Can be interpreted as the harmonic mean of the Precision and Recall. Best value is 1 and worst value is 0. Since it take Recall and Precision into account we used the F1 score to evaluate our Machine Learning model's performances.
- **Cohen's Kappa:** Adjusts the accuracy of a model by the probability of a random agreement. Therefore, it is more stable against imbalanced data sets, which we have. So, we also used Cohen's Kappa to evaluate our Machine Learning models. 
- **Log Loss:** For example, it is used as a metric in logistic regression and neural networks. Since we did not implement a Multi Layer Perceptron or a similar classifier we did not use it.
- **ROC AUC:** Measures the Area Under the ROC Curve. So, it can be interpreted as the probability that the model ranks a random positive example higher than a random negative example. 

### Baselines

Baseline classifiers are used to have a minimum starting point that the machine learning models must exceed in order to add some value.
We implemented five baseline classifiers namely the majority vote classifier, the random vote classifier, the always 'True' classifier, the always 'False' classifier, and the label frequency classifier.

- **Majority vote classifier:** Returns always the label of the majority class which is in our case `False`. So, it is the same as the always 'False' classifier.
- **Random vote classifier:** Returns uniform distributed labels. In our case it would return 50 % `True` and 50 % `False`.
- **Always 'True' classifier:** Returns always the label `True`.
- **Always 'False' classifier:** Returns always the label `False`. Is the same as the majority vote classifier in our case.
- **Label frequency classifier:** Returns labels based on the label frequency in the training data. For our data it is ~90 % `False` and ~10 % `True`.

### Baseline results

| Classifier         | Data       | Accuracy     | F1 score   | Cohen's Kappa |
| ------------------ | ---------- | -----------: | ---------: | ------------: |
| Majority vote      | Training   | **0.9058**   | 0.0000     | 0.0000        |
|                    | Validation | **0.9058**   | 0.0000     | 0.0000        |
| Random vote        | Training   | 0.5000       | 0.1578     | -0.0007       |
|                    | Validation | 0.4993       | 0.1601     | 0.0018        |
| Always 'True'      | Training   | 0.0942       | **0.1721** | 0.0000        |
|                    | Validation | 0.0942       | **0.1721** | 0.0000        |
| Always 'False'     | Training   | **0.9058**   | 0.0000     | 0.0000        |
|                    | Validation | **0.9058**   | 0.0000     | 0.0000        |
| Label frequency    | Training   | 0.8296       | 0.0943     | **0.0002**    |
|                    | Validation | 0.8307       | 0.0995     | **0.0061**    |

Because we have ~90 % negative labeled data points in our data sets it is not surprising that the majority vote and always 'False' classifier have the best accuracy. 
Also, the label frequency classifier has a high accuracy with ~83 %. 
With respect to the F1 score, the always 'True' classifier performs best with a value of 0.1721. 
For the last metric, the Cohen's Kappa score, the label frequency classifier performs best even if it is not a good score.

The results show that the accuracy is not a good evaluation metrics for our imbalanced data set.
It also shows that the F1 score and Cohen's Kappa score are more appropriate to evaluate the Machine Learning model performances.
Therefore, we just will use the two evaluation metrics in the classification step.
Furthermore, the results show that the overall best baseline is the label frequency classifier which we will try to overcome.

---

## Preprocessing

In the following we describe all preprocessing steps that we have implemented. 

### Language filter

The language filter can be used to remove tweets from the data set that are in a different language than the target language.

####  Design Decisions

Since not all tweets in the dataset are written in English we had to decide whether we remove or translate them.
To decide which option to choose, we took a look at the language distribution of the tweets.

| Place | Language id | Number of tweets |
| ----: | ----------- | ---------------: |
| 1.    | EN          | 283.240           |
| 2.    | ES          | 3.492             |
| 3.    | FR          | 3.287             |
| 4.    | DE          | 811              |
| 5.    | IT          | 748              |
| 6.    | IN          | 631              |
| 7.    | UND         | 546              |
| 8.    | NL          | 396              |
| 9.    | PT          | 389              |
| 10.   | TL          | 362              |

As the majority of our tweets are english tweets, we decided to remove the non-english tweets.
We had two possibilities to do so. 
First, we could use the column `language` from the dataset. 
Second, we could use some external service or class to determine the language from the `tweet` column.
Although we have found that the language in the `language` column is not always correct, we have used it because the algorithms that determine the language of a text do not always provide the correct language either.

#### Implementation Details

The `LanguageFilter` class does not create a new column like the other preprocessing steps but removes some rows from the data set.
Therefore, we could not use the `Preprocessor` class.
Instead, we implemented a base class for filtering, i.e. the `Filter` class, that removes certain indices from a data frame.

#### Results

With our data the `LanguageFilter` was able to remove 12.571 samples from the 295.811 samples in our data set. 
So it kept ~96 percent of the data or to be specific 283.240 data samples which is more than enough to train a classifier.

#### Interpretation

The removal of non-english tweets is a crucial point. 
If you want to use not only metadata of the tweet to predict its virality, having a single language in your data set reduces the complexity of all further steps in the pipeline, e.g. sentiment analysis.

### Removing punctuation

The punctuation remover removes all the punctuation from a tweet. 
It was implemented during the online lectures.

Sometimes the punctuation of a text does not contain much information. 
But since we want to use the punctuation of a tweet as a feature we did not use this preprocessing step.

### Removing stop words

The stop word remover removes all english stop words from either the raw tweet or the tokenized tweet.

#### Design Decisions

Often stop word do not contain much information. 
Therefore, it is common to remove them. 
Because we also want to count the number of words in a tweet that are somehow meaningful we also remove them.

#### Implementation Details

We use the english stop words from the `nltk` package. 
The stop words include for example pronouns (e.g. I, me, my), common verbs (e.g. be, have, will), prepositions (e.g. over, under), and many more frequently used words.

#### Results

 > **Example sentence:** Prof. Jeremy Petranka explains how #dataanalysis can identify policies that contribute to systemic #discrimination and #racism, and how data can also help change policies to make them more equitable

 > **Sentence after stop word removal:** Prof. Jeremy Petranka explains ~~how~~ #dataanalysis ~~can~~ identify policies ~~that~~ contribute ~~to~~ systemic #discrimination ~~and~~ #racism, ~~and how~~ data ~~can~~ also help change policies ~~to~~ make ~~them more~~ equitable

### Tokenize the tweet

The tokenizer splits the input text in its word tokens and returns them as a list. 

#### Design Decisions

To simplify the feature extraction we split the text contained in the `tweet` column into its word tokens.
The list of word tokens is then used in the feature extraction step to determine the word count for example.

#### Implementation Details

We again used the `nltk` package to split the tweets first into sentences and then into words.

#### Results

The sentence `"This is an example sentence"` is split by the tokenizer in the following tokens `["This", "is", "an", "example", "sentence"]`.

---

## Feature Extraction

### Media

#### Design Decisions

The first kind of features are related to media.
Media can be either an internal media type like videos or pictures or an external media type like links to external websites.
All media types can provide an additional source of information besides the relatively short tweets.
Therefore, they could also influence the vitality of a tweet.

Thus, we extracted three media-related features from our data.
We counted the number of photos that are posted with a tweet and the number of links that are contained in the tweet.
As both information are stored in a list in a separate data cell (i.e. `urls` and `photos`) we just had to count the elements in these list.
Therefore, we used a base class that counts the elements in a list and two derived classes for each feature that defines the name of the input and output columns. 
The third feature is extracted from the `video` column.
Because the values of the column are either 0 or 1 we can use the values directly.

#### Results

The following figure shows the relative frequencies of how many images are posted with a tweet.
About 59% of tweets do not contain a photo.
Of the remaining 41 % tweets, mostly only one photo is posted. 
The maximum number of photos is four which is also the maximum of allowed photos per tweet ([Source: Twitter - Upload media](https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/uploading-media/media-best-practices)).

![Shows the relative frequencies of photos posted with a tweet.](figures/photos.png "Relative frequencies of photos")

In contrast to the number of photos posted with a tweet the most tweets (~81 %) contain at least one URL where one URL is also the most common one with ~75 %.
Around 4 % of the tweets contain two urls and the remaining 2 % of the tweets contain three to eight URLs.

![Shows the relative frequencies of URLs contained in the tweets.](figures/urls.png "Relative frequencies of URLs")

Most tweets (~58 %) do not contain a video as the following figure shows. 
The other ~42 % contain one video which is also the maximum of allowed videos ([Source: Twitter - Upload media](https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/uploading-media/media-best-practices)).

![Shows the relative frequencies of videos posted with a tweet.](figures/video.png "Relative frequencies of videos")

#### Interpretation

Since the length of a tweet is limited to 280 characters and in the past even to only 140 characters ([Source: Twitter - Counting characters](https://developer.twitter.com/en/docs/counting-characters)), it is important to use this limited length as well as possible.
So, the virality of a tweet will probably not benefit from a high number of URLs.
However, providing one URL could be beneficial as it can lead the user to further information about the content of the tweet.
Also, it might be important to let the tweet take up as much space as possible in the feed, as it is more visible that way.
This can be achieved by adding a video or photos. 
But since you can either add a video or photos ([Source: Twitter - Upload media](https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/uploading-media/media-best-practices)) and the majority of the tweets (~83 %) have added a video or photos this features may carry not too much information about the tweet's virality.

### Tweet length

#### Design Decisions

The second kind of features are related to the tweet's length. 
As already mentioned before, the length of a tweet is limited to 280 characters ([Source: Twitter - Counting characters](https://developer.twitter.com/en/docs/counting-characters)). 
So, the number of characters used in a tweet can maybe beneficial to predict its virality. 
To extract the character length of a tweet we just had to count the number of characters.
Furthermore, we also wanted to include the number of words as a feature.
We counted all word tokens as well as all word tokens without stop words using the previous implemented `ListCounter` class.
By including both values as features we hoped that the classifier could learn something about how meaningful a tweet is which hopefully improves the virality prediction.

#### Results

In the following figure, only tweets with less than 341 characters are considered.
So, a total of 390 tweets were excluded.
Considering that a tweet can only be 280 characters long, it is still strange why some tweets have more characters.
A reason for this could be that some entity objects impact the length of a tweet ([Source: Twitter - Counting characters](https://developer.twitter.com/en/docs/counting-characters)).

The histogram shows clearly that in the past the maximal number of characters was limited to 140 characters. 
The number of tweets that contain a certain number of characters increases until the maximum of ~38.000 tweets is reached for tweets that are around 140 characters long. 
Afterwards the number of tweets drops and is relatively uniform distributed around 6.000 tweets with small peaks at 160, 280, and 310 characters.

![Shows the histogram of the character length of a tweet.](figures/char.png "Histogram of character length")

For the next figure, only tweets with less than 71 tokens are considered.
So, 337 tweets are not included.

The figure shows two histograms one for the number of all word tokens in the tweet and the other for the number of word tokens exclusive stop words. 
Again one can see that the character limit was increased over time.

![Shows the histogram of the number of tokens in a tweet.](figures/tokens.png "Histogram of number of tokens")

The last figure shows the difference between the number of tokens with and without stop words. 
The size of each point in the scatter plot shows how many of the tweets have a certain difference relative to the number of tokens with stop words.
With increasing number of tokens also the difference range increases. 

![Shows a scatterplot of the number of tokens in a tweet in relation to the difference to the number of tokens without stop words.](figures/tokens_diff.png "Difference between the number of tokens with and without stop words.")

#### Interpretation

Probably these features are not very useful for classification for two reasons. 
First, we did not include a feature which indicates whether the tweet was posted before or after the character length increased.
Another solution for that problem would have been to divide the number of characters by 140 or 280 depending on the character liming at the posting time.
Second, it would have made more sense to include the difference between the tokens with and without stop word rather than both numbers independent.
In this case, the classifier would not have to learn the dependency first.

### Time

#### Design Decisions

Another useful feature type could be time-related features. 
Because there are time slots where more users are online and probably the interaction rate with a newly posted tweet influences also if the tweet is shown to other users, the posting time is important for the tweet's virality.
Therefore, we decided to implement two time-related features. 
The first looks at the time, categorize the posting time in one of four categories, and returns the one-hot encoded category as feature, where the four categories are night (00:00-5:59), morning (06:00-11:59), afternoon (12:00-17:59), and evening (18:00-23:59).
The second takes the date, looks the weekday up, and returns the one-hot encoded weekday as feature.

#### Results

The following figure shows the relative frequencies of the time categories. 
The most tweets are posted in the evening (~38.5%), followed by tweets that are posted in the night (~31.5 %), in the morning (~16.5 %), and in the afternoon (~13.5 %).

![Shows the relative frequencies of the time categories.](figures/time.png "Relative frequencies of the time categories")

Looking at the class distribution for each time category one can see that the morning class contain ~15 % viral tweets.
It is followed by the night category with ~9.5 %, the evening category with ~8 %, and the afternoon category with ~7 %.

![Shows the class distribution of each time category.](figures/dist_for_each_time.png "Class distribution for each time category")

The relative frequencies of the weekdays are shown in the next figure. 
Most tweets are posted on the days from Friday to Monday.
After Monday the posting rate drops until it reaches it low at Thursday, where the posting rate on Thursday is about half the posting rate from Friday to Monday.

![Shows the relative frequencies of the weekdays.](figures/weekday.png "Relative frequencies of the weekdays")

In the following figure the class distribution for each weekday is shown. 
Again one can see that in comparison to the number of tweets per weekday the part of viral tweets is bigger if the number of tweets is lower.
Therefore, Thursday contains the most viral tweets with ~14 % followed by Wednesday (~11.5 %) and (~10.5 %).
The days with the most tweets, i.e. Friday, Saturday, Sunday, and Monday, just contain ~8 % viral tweets each.

![Shows the class distribution of each weekday.](figures/dist_for_each_weekday.png "Class distribution for each weekday")

#### Interpretation

Time-related features are probably very important for the virality prediction.
If more users are active at the posting time it could be more likely that a tweet goes viral. 
As it can be seen in the data a good time to post something could be Thursday morning.

### Sentiment

#### Design Decisions

Since it is likely that the tone of a tweet influences its virality, we decided to extract the sentiment of the tweet.
The sentiment is split into the polarity, i.e. whether a tweet is positive or negative connoted, and the subjectivity, i.e. whether it expresses a person's opinion or describes the world as it is.

#### Results

The scatter plot shows the polarity vs. the subjectivity of the tweets with respect to the assigned labels. 
It also visualizes the number of tweets that have a certain combination of polarity and subjectivity by the size of the marker.
Overall, it can be said that the most tweets have a positive polarity and are rather objective. 
Unfortunately, it seems that the viral and non-viral tweets are evenly distributed.

![Shows the polarity vs subjectivity of the tweets with respect to the assigned labels.](figures/sentiment.png "Polarity vs. Subjectivity")

#### Interpretation

Since it seems that viral and non-viral tweets are evenly distributed - except for a few exceptions - the sentiment will probably not help to predict the virality of a tweet.

### Mentions & Hashtags

#### Design Decisions

As the number of persons mentioned in a tweet and the number of used hashtags influences somehow the discoverability we include both numbers as features.
The former one influences the discoverability because the persons that are mentioned gets a notification and will probably interact with the tweet.
Because of the later one a tweet can be found better via the hashtag and is shown to all persons that follow a certain hashtag.
Both feature extractors inherit again from the `ListCounter` class and use the column `mentions` or `hashtags` as input data.

#### Results

In most tweets no person is mentioned (~56.5 %). 
However, ~29 % mention one person and ~9 % mention two persons. 
There are also even tweets iin which 18 persons are mentioned
So, in theory a person can mention many persons but in practice it seems that most persons just mention up to two twitter users.

![Shows the relative frequencies of the number of mentions.](figures/mentions.png "Relative frequencies of the number of mentions")

Looking at the class distribution for each number of mentions it seems that it is most promising to mention many people.
Further, mentioning just one or even no person is better than mentioning two or three people. 

![Shows the class distribution for each mention.](figures/dist_mentions.png "Class distribution for each mention")

In our data the range for the number of hashtags used in a tweet reaches from zero to 29. 
With ~33.5 % the tweets without any hashtags is the biggest group.
For an increasing number of hashtags used the number of tweets decreases. 

![Shows the relative frequencies of the number of hashtags.](figures/hashtags.png "Relative frequencies of the number of hashtags")

The class distribution for each number of hashtags shows that it could be beneficial to use more hashtags.
The share of viral tweets increases from ~6 % for tweets with one hashtag to ~18 % for tweets with seven hashtags.
Afterwards the share decreases.

![Shows the class distribution for each hashtags.](figures/dist_hashtag.png "Class distribution for each hashtag")

#### Interpretation

Both features seem to carry some information about the tweet's virality.
So, they are probably useful to predict the virality of a tweet. 
In general, it seems to be a good idea to use many hashtags to increase the audience of your tweet.

### Character level features

#### Design Decisions

The last feature types are character level features. 
We counted the number of punctuation marks and the number of capital letters.
From both of them we promised ourselves that they would be beneficial for the virality prediction since they probably influence the visibility of a tweet.
For example, if a person uses multiple exclamation points, it may indicate that the tweet is important to them.
The same can be true when a person writes a tweet in all caps.

Since both features are counts on the character level and depend on a condition we implemented the base class `ConditionalCharCounter` from which both feature extractors inherit.

#### Results

The histogram shows the distribution of the number of punctuation marks in a tweet. 
For visibility reasons all tweets with more than 50 punctuation marks were excluded (34 tweets).

![Shows the histogram of the number of punctuation characters.](figures/punc.png "Histogram of the number of punctuation characters ")

The next histogram shows the distribution of the number of capital letters in a tweet.
For visibility reasons all tweets with more than 60 capital letters were excluded (248 tweets).

![Shows the histogram of the number of capital characters.](figures/cap.png "Histogram of the number of capital characters")

#### Interpretation

Both histograms do not show a clear number range where the tweet is much more likely to become viral.
So, probably these features will not be as useful as others.

---

## Dimensionality Reduction

We have a total of 14 feature extractors.
Twelve of them return a single value, one is a one-hot encoding of four categories, and the last is a one-hot encoding of seven categories.
So, each tweet can be represented as an array of length 23 (= 12 * 1 + 1 * 4 + 1 * 7). 
Since this is not a lot we decided to not implement a dimensionality reduction technique.

---

## Classification

Classification is the process of predicting the class of given data points. The task of classification predictive modeling 
is an approximation to a mapping function (f) from input variables (X) to discrete output variables (y).
For our pipeline we implemented four classifiers: K-Nearest Neighbor classifier, Support Vector machine classifier, Random Forest classifier 
and Gaussian Naive Bayes classifier.

### K-Nearest Neighbor (KNN)
The KNN classifier classifies a new instance based on its closest neighbors in the feature space.

#### Results
We performed a systematic grid search to find the optimal value for the `n_neighbors` parameter. 
For evaluation we took the Cohen's Kappa and the F1 score into account. 

![Shows the results of the KNN hyperparameter optimization.](figures/KNN-mlflow.png "MLflow results of KNN hyperparameter optimization")

The results show that the best value for `k` in our case is 1.


### Support Vector Machine (SVM)
Classifies a new instance based on its largest-margin hyperplane.

#### Results
We performed a systematic grid search to find the optimal value for the `C` and the `kernel` parameter. 
For evaluation we took the Cohen's Kappa and the F1 score into account. 

![Shows the results of the SVM hyperparameter optimization.](figures/SVM-mlflow.png "MLflow results of SVM hyperparameter optimization")

The results show that the best values in our case are: `C = 10.0` and `kernel = sigmoid`.

Some run combinations have been killed by the grid for exceeding wall time, since the time to compute scales with the size of the dataset.
It would be recommended to test the LinearSVC classifier in future work, since it is more suitable for large datasets.


### Random Forest (RF)
Uses the output of multiple weaker classifiers in order to make a classification decision.

#### Results
We perfomed a systematic grid search to find the optimal value for the following parameters: `n_estimators, criterion` and `class_weights`.
For evaluation we took the Cohen's Kappa and the F1 score into account. 
![Shows the results of the RF hyperparameter optimization.](figures/RF-mlflow.png "RF results of SVM hyperparameter optimization")

The results show that the best values in our case are: `n_estimators = 200`, `criterion = entropy` and `class_weights = None`.
It should be noted that runs with higher values for `n_estimators` (values of 500 and 1000) were killed due to memory excess in the grid.


### Gaussian Naive Bayes (NB)
NB uses conditional probabilities for computing the probability of a given data point belonging to a given class.

#### Results
For NB, we did not perform hyperparameter optimization. The Naive Bayes GaussianNB classifier contains two parameters, `priors` and `var_smoothing`.
The priors are being adjusted according to the data. The second parameter takes the portion of the largest variance of all features that is added to variances for calculation stability.
We decided to keep the default values of the GaussianNB classifier.

---


## Final Evaluation and Interpretation

Given the results of our grid search, our best performing candidate classifier was the Random Forest Classifier with 200 estimators, the split criterion being the entropy and the class weights set to `None`.
With this setup we achieved a Cohen's Kappa of ~0.138, an F1-score of ~0.159 and an accuracy of ~0.9086 on the validation set.
The final evaluation on the test set confirms the validation results with a Cohen's Kappa of ~0.148, an F1-score of ~0.1698 and an accuracy of ~0.9095.

These results show that, although not outperforming our baseline by large on accuracy, our Random Forest classifier was able to extract - at least to some extent - 
certain relevant features and patterns from a tweet for a correct positive (or _true positive_) classification. This can be drawn from the F1-scores where our baseline - the "always false" classifier - gets the value 0 due to its inherent
missing true positive classifications.

---


