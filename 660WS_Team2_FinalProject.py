

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pathlib import Path

#defining the function which takes filename as input and returns a list of reviews and labels
def loadData(filename):
    fname=Path(filename)
    reviews=[]
    labels=[]
    f=open(fname,encoding = 'utf-8')
    for line in f:
        if len(line.strip().rsplit('\t',1))==2:
            review,rating=line.strip().rsplit('\t',1)
            reviews.append(review.lower())    
            labels.append(int(rating))
    f.close()
    return reviews,labels

#defining the function which takes filename as input and returns a list of reviews
def loadUnlabelData(filename):
    fname=Path(filename)
    reviews=[]
    f=open(fname,encoding = 'utf-8')
    for line in f:
        review=line.strip()
        reviews.append(review.lower())
    f.close()
    return reviews

# Give the path for the training data 
rev_train,labels_train= loadData("train_final.txt")

# Give the path for the testing data 
rev_test=loadUnlabelData("test_final.txt")

# Logistic Regression and tfidf pipeline
tvc_pipe = Pipeline([
 ('tvec', TfidfVectorizer(max_df=0.6,sublinear_tf=True)),
 ('logreg', LogisticRegression())
])

tvc_pipe.fit(rev_train, labels_train)

# printing the predicted labels
predicted= tvc_pipe.predict(rev_test)
np.savetxt('predicted_labels.txt',predicted,fmt="%d")                          

