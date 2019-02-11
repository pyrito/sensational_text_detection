import tensorflow as tf
from tensorflow.contrib import keras
import pattern.en
import sys
from pattern.en.wordlist import PROFANITY
import pickle
import string
import collections
import numpy as np
# nltk.download('wordnet')

class Classifier:
    def __init__(self):
        # ***** Setup vectorizers for features that don't naturally produce numeric values (POS and syntax)
        with open("posvect.pkl", "rb") as pik:
            self.pos_vectorizer = pickle.load(pik)
            print("pickle preloaded for pos vectorizer")

        with open("chunkvect_count.pkl", "rb") as pik:
            self.chunk_vectorizer = pickle.load(pik)
            print("Pickle preloaded for chunk vectorizer")
        try:
            print("Loading model")
            self.saved_model = keras.models.load_model('sensational_detector_model.h5')
            self.saved_model._make_predict_function()
            self.graph = tf.get_default_graph()
        except FileNotFoundError:
            print("Model does not exist")
            exit(1)

# Define functions for feature tagging the text
    def __pos_tag(self, title, text):
        """
        Uses NLTK to POS tag the titles and the texts of each text
        :param title:
        :param text:
        :return: (vectorized POS for title, vectorized POS for text)
        """
        text_words = str.split(text)
        stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        text_words = list(filter(lambda x: x.lower() not in stop and x.lower() not in string.punctuation, text_words))
        # tagged_text = [" ".join(x[1] for x in pattern.en.tag(text_words))]
        tagged_text = []
        for x in text_words:
            try:
                o = pattern.en.tag(x)
                if isinstance(o[0][1], str):
                    tagged_text.append(o[0][1])
                    # print(o)
            except Exception as e:
                print(e, x)
                # pass
        tagged_text = [' '.join(tagged_text)]
        title_words = str.split(title)
        title_words = list(filter(lambda x: x.lower() not in stop and x.lower() not in string.punctuation, title_words))
        # tagged_title = [" ".join(x[1] for x in pattern.en.tag(title_words))]
        tagged_title = []
        for x in title_words:
            try:
                o = pattern.en.tag(x)
                if isinstance(o[0][1], str):
                    tagged_title.append(o[0][1])
                    # print(o)
            except Exception as e:
                print(e, x)
                # pass
        tagged_title = [' '.join(tagged_title)]
        return self.pos_vectorizer.transform(tagged_title), self.pos_vectorizer.transform(tagged_text)
    def __profanity_scan(self, title, text):
        """
        Uses Pattern to count the number of profane words in the text
        :param title:
        :param text:
        :return: return tuple with frequency of profane words in title and text (# of profane in title, # of profane in text)
        """
        profane_list = set(PROFANITY)
        text_words = str.split(text)
        text_count = 0
        title_count = 0
        for word in text_words:
            if word.lower() in profane_list:
                text_count += 1
        for word in title.split():
            if word.lower() in profane_list:
                title_count += 1
        return title_count, text_count
    def __sentiment_scan(self, title, text):
        """
        Uses Pattern to get the polarity and subjectivity score provided by the pattern sentiment module
        :param title:
        :param text:
        :return: Pair of tuples with polarity and subjectivity scores of title and text ((title polarity, title subjectivity), (text polarity, text subjectivity))
        """

        return (pattern.en.sentiment(title), pattern.en.sentiment(text))
    def __sent_len(self, title, text):
        """
        Length of title and text
        :param title:
        :param text:
        :return: (length of title, average length of sentences in text)
        """
        total = 0
        text_sent = str.split(text)
        for sent in text_sent:
            total += len(str.split(sent))
        return (len(str.split(title)), total / len(text_sent))
    def __syntax(self, text):
        """
        Use the pattern sentence tree parsing tool to split up the sentence into its chunk permutation
        :param title:
        :param text:
        :return: (chunk permutations of each type of the entire text)
        """
        s = pattern.en.parsetree(text, relations = True, lemmata = True)
        text_chunks = []
        for sentence in s:
            out = ""
            for chunk in sentence.chunks:
                out += str(chunk.type)
            text_chunks.append(out)
        text_chunks_out = [" ".join(text_chunks)]
        return (self.chunk_vectorizer.transform(text_chunks_out),)
    def __emphasis(self, title, text):
        """
        Determine how many uses of exclamations or questions
        :param title:
        :param text:
        :return: (Punctuation in title, punctuation in text)
        """
        title_counts = collections.Counter(title)
        text_counts = collections.Counter(text)
        text_count = 0
        title_count = 0
        exclamatory = ('?', '!')
        for k in exclamatory:
            if title_counts[k] is not None:
                title_count += title_counts[k]
            if text_counts[k] is not None:
                text_count += text_counts[k]
        return text_count, title_count
    def __capitals(self, title, text):
        """
        Scan through the text and deteremine how many words are all capital letters
        :param title:
        :param text:
        :return: (# of words in all CAPS of title, # of words in call CAPS of text)
        """
        text_words = str.split(text)
        text_count = 0
        title_count = 0
        for word in text_words:
            if word.isupper():
                text_count += 1
        for word in title.split():
            if word.isupper():
                title_count += 1
        return title_count, text_count

    def classify(self, title, text):
        pos = self.__pos_tag(title, text)
        prof = self.__profanity_scan(title, text)
        sent = self.__sentiment_scan(title, text)
        length = self.__sent_len(title, text)
        synt = self.__syntax(text)
        emp = self.__emphasis(title, text)
        capt = self.__capitals(title, text)

        pos_title = pos[0].todense()[0].tolist()[0]
        pos_text = pos[1].todense()[0].tolist()[0]
        prof_title = [prof[0]]
        prof_text = [prof[1]]
        sent_title = [sent[0][0], sent[0][1]]
        sent_text = [sent[1][0], sent[1][1]]
        len_title = [length[0]]
        len_text = [length[1]]
        synt_text = synt[0].todense().tolist()[0]
        emp_title = [emp[0]]
        emp_text = [emp[1]]
        cap_title = [capt[0]]
        cap_text = [capt[1]]

        output_vector = [pos_title, pos_text, prof_title, prof_text, sent_text, sent_title, len_title, len_text,
                         synt_text, emp_title, emp_text, cap_title, cap_text]

        # print(self.pos_vectorizer.vocabulary_)
        # print(self.chunk_vectorizer.vocabulary_)
        # print(output_vector)
        vect = [keras.preprocessing.sequence.pad_sequences(output_vector,
                                                          value=0,
                                                          padding='post',
                                                          maxlen=128)]

        sample = np.array(vect)

        with self.graph.as_default():
            result = self.saved_model.predict(sample)
        print (result, file=sys.stderr)
        if result[0][0] > .6:
            return 1
        else:
            return 0
        # return result[0]