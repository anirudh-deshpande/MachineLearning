from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

documents = ['Hello! Its me',
             'How are you doing? Are you free today?',
             'You are really cute :) :P',
             'My number is 7411027576']

count_vector = CountVectorizer()
print count_vector

# It first gets all the features at one place
count_vector.fit(documents)
print count_vector.get_feature_names()

# Transforms the documents into frequency vectors
# We can also input a DataFrame into transform.
# Eg. training_data = CountVectorizer.fit_transform(X_train) -> outputs scipy matrix
doc_array = count_vector.transform(documents).toarray()
print doc_array

# Frequency matrix
frequency_matrix = pd.DataFrame(doc_array,
                                columns=count_vector.get_feature_names())

print frequency_matrix