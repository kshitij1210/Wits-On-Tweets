# Wits-On-Tweets

## Requirements

1.) Python 3.3 or above

2.) Keras with Tensorflow backend

3.) Keys for the Twitter Tweepy API (Tweepy should be installed first)

4.) Numpy 

5.) Pandas

6.) NLTK completely installed

7.) ggplot

8.) Matplotlib

## Motivation

In all our daily lives we take decisions which are highly motivated by people around us. Nowadays the decisions we take are not only dependent on the reviews of people closest to us, but also on the reviews from everyone around the globe. Websites like Twitter, Facebook help us to quickly gather a lot of information about any particular topic. Our project aims to help individuals gather live information about any topic on Twitter based on any keyword and predict the sense of the tweets to make better real-time decisions.

## What We Did

We implemented a model that can accurately classify Twitter messages as positive or negative based on probabilities. Our hypothesis is that we can obtain high accuracy on classifying sentiment in Twitter messages using deep learning techniques. We used 2 models - 

1.) Multilayered Perceptron

2.) Long-Short Term Memory(LSTM)

We also did negation handling whose procedure has been described below.

## Block Diagram of Our Work Flow

![alt text](https://github.com/kshitij1210/Wits-On-Tweets/blob/master/images/block%20diagram.jpg)

## Negation Handling
The problem to handle negations in sentences is a tough task in itself while using the bag of words model. The bag of words model does not consider the order of tokens for classification. Another problem is that while removing stopwords, words like not and no are removed because they convey no emotion in itself. To properly classify the sentences with tokens like no, not, n't and never, the approach used is to concatenate the token with the next one. This makes it entirely a new token and this token can be used in our bag of words model.

## Our Result

We compared two different approaches to tackle sentiment analysis problem using Neural Networks. Of the two models we implemented, both the accuracy are comparable. We achieved 77.7% accuracy in case of Feed Forward with Backpropagation model and a slightly less 76.5% accuracy in case of RNN-LSTM. It was noticed that increasing the size of the dataset and the number of words included in the dictionary also increases our accuracy. But due to the hardware re-strictions, this was the maximum accuracy achieved.

## Sentiment Analysis Package for Multilayered Perceptron
We have converted our sentiment analysis module to a package which can be used by anyone to test given string. For doing so we saved the structure of model and dictionary of words in JSON file and the weights and biases of the nodes in h5 file format. This makes it exible enough for end users and developers to use our work and also improve it.

## Some Examples

![alt text](https://github.com/kshitij1210/Wits-On-Tweets/blob/master/images/Test_string_positive1.PNG)

![alt text](https://github.com/kshitij1210/Wits-On-Tweets/blob/master/images/Test_string_positive2.PNG)

![alt text](https://github.com/kshitij1210/Wits-On-Tweets/blob/master/images/Test_string_negative1.PNG)

![alt text](https://github.com/kshitij1210/Wits-On-Tweets/blob/master/images/Test_string_negative2.PNG)

## Live Plot of Tweets

Using the Twitter API, the tweets are fetched from Twitter based on any keyword, between the given dates and stored the sentiment data in a file. By reading the last 200 tweets from the file, we plot the graph of sentiment about tweets having the given keyword. The result we obtained for Microsoft are depicted below:

![alt text](https://github.com/kshitij1210/Wits-On-Tweets/blob/master/images/live_data_plot.PNG)
