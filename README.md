# sensational_text_detection
A web-app that will detect if the input text is sensationalized or not.
Hey! This website was built as an extension of a project that was developed for our Computational Semantics class at UT Austin (LIN 350). Our machine learning model was inspired by a Github project, which attempted to achieve the same goal that we had: to determine whether a news article was sensationalized.

The initial scraping of website information is done by Newspaper, a powerful tool used to extract metadata from news sites without having to do the dirty work ourselves. It is very accurate and powerful and allowed the webapp to easily parse out the title and text information that is needed for our model.

The model uses text analysis tools such as NLTK and Pattern to extract features from the text. We look at features including sentence structure, part-of-speech usage, sentence length, sentiment, punctuation, etc. to input into a neural network and produce a classification for the article as sensationalized or not sensationalized. While it would seem intuitive to go ahead and focus on the words used in sensational, bias pieces of text, we decided to try experimenting with focusing on semantical usage instead (since most sensational text is subtle and not explicit.)

Our model architecture works as follows (using NLTK, Pattern and Tensorflow):

Feature tagger that takes raw text -> numeric data vector
Neural network architecture: one single convolutional layer, three fully connected dense layers with 16 neurons, fully connected output layer with a single output neuron.
We trained this model on 16,000 training articles and 2,000 testing articles. Data set was extracted from the News Audit project. Below you can see the training accuracy of our model: 
After training our model to be around 95% accurate with the training set, we tested our model with the test set and got an accuracy of around 96%, a 3-4% increase over the other models that used this same data set.

Important Note:
Although there is often a correlation between sensationalism and "fake news", our model is not a fake news detector. It does NOT fact check the articles. In fact, it never looks at the words individually in terms of what they mean. All of our analysis is purely based on the semantics and linguistic structure of the text. For example, articles from sites such as The Onion, which writes fake news for satirical purposes, can be noted as "not sensationalized" due to the lexical techniques used to maintain a completely serious, objective tone.
