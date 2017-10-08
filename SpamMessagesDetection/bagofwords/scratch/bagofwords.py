import string
from collections import Counter
import pprint

documents = ['Hello! Its me',
             'How are you doing? Are you free today?',
             'You are really cute :) :P',
             'My number is 7411027576']

# Step 1: Lower case
lower_case_documents = []
for doc in documents:
    lower_case_documents.append(doc.lower())
# print lower_case_documents


# Step 2: Remove punctuation
# print string.punctuation
# https://www.tutorialspoint.com/python/string_translate.htm
# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
rmv_punctuation_lc_documents = []
for doc in lower_case_documents:
    rmv_punctuation_lc_documents.append(doc.translate(string.maketrans("",""), string.punctuation))
    # string.maketrans("","") argument is for dummy purpose
# print rmv_punctuation_lc_documents


# Step 3: Tokenize
pre_processed_docs = []
for doc in rmv_punctuation_lc_documents:
    pre_processed_docs.append(doc.split(' '))
# print pre_processed_docs


# Step 4: Count the frequencies
frequency_count_list = []
for arr in pre_processed_docs:
    frequency_count = Counter(arr)
    frequency_count_list.append(frequency_count)
# print frequency_count_list
# pprint.pprint(frequency_count_list)
