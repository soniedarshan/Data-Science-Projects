import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import re
random.seed(0)
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def words_cleanup(data, stopwords):
"""
Removes stop words from the data
"""
    result = list()
    for item in data:
        # remove stopwords
        temp = [word for word in item if not word in stopwords]

        # insert to result
        result.append(temp)

    return result

def get_bag_of_words(data, final_word_vector):
"""
Creates binary feature vectors. If word is present, set boolean value equal to 1 else set to 0.
"""
    result = list()
    for line in data:
        lineSet = set(line)
        currVect = list()
        for word in final_word_vector:
            if word in lineSet:
                currVect.append(1)
            else:
                currVect.append(0)

        result.append(currVect)  
    return result          

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    train_pos_vec_clean = words_cleanup(train_pos, stopwords)
    train_neg_vec_clean = words_cleanup(train_neg, stopwords)

    pos_count_dict = dict()
    neg_count_dict = dict()

    pos_count = len(train_pos_vec_clean)
    neg_count = len(train_neg_vec_clean)

    allowed_pos_count = int(0.01 * pos_count)
    allowed_neg_count = int(0.01 * neg_count)

    #print allowed_neg_count, allowed_neg_count

    #print pos_count, neg_count

    for pos_line in train_pos_vec_clean:
        pos_line_set = set(pos_line)
        for word in pos_line_set:
            if word in pos_count_dict:
                pos_count_dict[word] = pos_count_dict[word] + 1
            else:
                pos_count_dict[word] = 1    

    for neg_line in train_neg_vec_clean:
        neg_line_set = set(neg_line)
        for word in neg_line_set:
            if word in neg_count_dict:
                neg_count_dict[word] = neg_count_dict[word] + 1
            else:
                neg_count_dict[word] = 1 

    #print pos_count_dict
    #print neg_count_dict            
    
    final_word_vector = list() # Set for fatser searching
    total_word_set = set(pos_count_dict.keys() + neg_count_dict.keys())


    #print len(pos_count_dict.keys())
    #print len(neg_count_dict.keys())
    #print len(total_word_set)

    for word in total_word_set:

        if word in pos_count_dict and word in neg_count_dict:
            criteria_1 = pos_count_dict[word] >= allowed_pos_count or neg_count_dict[word] >= allowed_neg_count
        elif word in pos_count_dict:
            criteria_1 = pos_count_dict[word] >= allowed_pos_count
        elif word in neg_count_dict:
            criteria_1 = neg_count_dict[word] >= allowed_neg_count
        else:
            print "Aw Snap! Something is fishy !!!"       

        if word in pos_count_dict:
            word_pos_count = pos_count_dict[word]
        else:
            word_pos_count = 0

        if word in neg_count_dict:
            word_neg_count = neg_count_dict[word]
        else:
            word_neg_count = 0 

        criteria_2 = word_pos_count >= 2 * word_neg_count or word_neg_count >= 2 * word_pos_count   
        
        if criteria_1 and criteria_2:
            final_word_vector.append(word)    


    #print len(final_word_vector)
    #print final_word_vector
    #print len(set(final_word_vector))    

    #print total_word_list

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    
    train_pos_vec = get_bag_of_words(train_pos, final_word_vector)
    train_neg_vec = get_bag_of_words(train_neg, final_word_vector)
    test_pos_vec = get_bag_of_words(test_pos, final_word_vector)
    test_neg_vec = get_bag_of_words(test_neg, final_word_vector)

    #print train_pos_vec[1:3]

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def make_labels(data, label):
"""
Assigns labels
"""
    result = list()
    for i in range(0, len(data)):
        curr_label = label + str(i)
        result.append(LabeledSentence(data[i], [curr_label]))
    return result

def get_vectors(model, label, n):
    result = list()
    for i in range(n):
        result.append(model.docvecs[label + str(i)])
    return result    



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    labeled_train_pos = make_labels(train_pos, "TRAIN_POS_")
    labeled_train_neg = make_labels(train_neg, "TRAIN_NEG_")
    labeled_test_pos = make_labels(test_pos, "TEST_POS_")
    labeled_test_neg = make_labels(test_neg, "TEST_NEG_")
    #print labeled_train_pos

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
  
    train_pos_vec = get_vectors(model, "TRAIN_POS_", len(labeled_train_pos))
    train_neg_vec = get_vectors(model, "TRAIN_NEG_", len(labeled_train_neg))
    test_pos_vec = get_vectors(model, "TEST_POS_", len(labeled_test_pos))
    test_neg_vec = get_vectors(model, "TEST_NEG_", len(labeled_test_neg))

    #print model.docvecs["TRAIN_NEG_0"]
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    Y_np = np.array(Y)
    X_np = np.array(X)

    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X_np, Y_np)


    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters

    lr_model = LogisticRegression()
    lr_model.fit(X_np, Y_np)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    
    Y_np = np.array(Y)
    X_np = np.array(X)

    nb_model = GaussianNB()
    nb_model.fit(X_np, Y_np)

    lr_model = LogisticRegression()
    lr_model.fit(X_np, Y_np)


    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    predictions_pos = model.predict(test_pos_vec)
    #print predictions_pos[0:3]
    #sys.exit(0)
    for pred in predictions_pos:
        if pred == "pos":
            tp = tp + 1
        else:
            fn = fn + 1

    predictions_neg = model.predict(test_neg_vec)
    #print predictions_pos[0:3]
    #sys.exit(0)
    for pred in predictions_neg:
        if pred == "neg":
            tn = tn + 1
        else:
            fp = fp + 1            

    accuracy = float(tp + tn) / (tp + tn + fn + fp)       
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
