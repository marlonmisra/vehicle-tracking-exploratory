import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
import pickle

#load data
print("Starting to load data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
print("Finished loading training data")
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
print("Finished loading testing data")

clf = LinearSVC(C = 0.0005, class_weight={0: 1, 1: 1})
t=time.time()
print("Training samples: ", len(X_train))
clf.fit(X_train, y_train)
t2=time.time()
print("Time (s) to train model:", round(t2-t, 2))

pred = clf.predict(X_test)
testing_accuracy = accuracy_score(pred, y_test)
print("Test Accuracy of SVM: ", testing_accuracy)

pickle.dump(clf, open("classifier.sav", 'wb'))
print("Classifier saved")

