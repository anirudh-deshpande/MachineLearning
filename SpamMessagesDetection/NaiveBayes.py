import os
from dataset.SMSMessages import get_messages_df
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

messages_df = get_messages_df(os.path.abspath('dataset/SMSSpamCollection'))

'''
Split messages into training and testing samples
'''
X_train, X_test, Y_train, Y_test = train_test_split(messages_df['message'],
                                                    messages_df['label'],
                                                    random_state=1) # random_state If int, random_state is the seed used,
                                                                    # If None, np.random

# training_data & testing_data document-term-matrix (Scipy matrix)
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# print type(testing_data) Compressed sparse row matrix

# Fit training data into Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, Y_train)

# Predictions
predictions = naive_bayes.predict(testing_data)

print 'Accuracy: ' + format(accuracy_score(Y_test, predictions))
print 'Precision: ' + format(precision_score(Y_test, predictions))
print 'Recall: ' + format(recall_score(Y_test, predictions))
print 'F1 Score: ' + format(f1_score(Y_test, predictions))

message_2 = ["Congratulations! You have won 10000000 dollars, hurrah!"]
message_3 = ["Hey, how are you?"]

testing_data_message_2 = count_vector.transform(message_2).toarray()
predictions_2 = naive_bayes.predict(testing_data_message_2)
print predictions_2

testing_data_message_3 = count_vector.transform(message_3).toarray()
predictions_3 = naive_bayes.predict(testing_data_message_3)
print predictions_3