Spam detection using UCI SMS messages dataset. SMS messages retrieved from the dataset: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

The Bayes theorem calculates the probability of a certain event happening based on the joint
probabilistic distributions of certain other events.

Naive Bayes operates on Conditional independence of events. Hence the name 'Naive'
Eg. If Age and Sex are the factors influencing occurrence of an event (A bank robbery),
they are assumed to influence independently.

Bag of Words is the frequency of words in the test text data being analyzed. Order of occurrence of words is not considered as important (just for conditional independence to hold true).

Bayes Theorm:
P(A|B) = (P(A) * P(B|A) / P(B) ( '/P(B)' normalizes the terms)

Eg. P(Diabetes|test:Positive) = P(Diabetes) * P(test:positive|Diabetes) / P(test:positive))

For multiple features (considering the features are independent),

P(A | B,C) = P(A | B) * P (A | C).

Eg. P(Good_Food | Taste:Yummy, Restaurant:Clean) = P(Good_Food | Taste:Yummy) * P(Good_Food | Restaurant:Clean)

Validations:
- Accuracy: How often the classifier makes the correct prediction (Comparing it to the actual predictions).
- Precision: What portion of messages classified as spam were actually spam [TP / TP + FP]
- Recall: What percentage of messages that actually were spam are classified as spam [TP / TP + FN]

 Note: For prediction of sick people,
 1. True Positive: Sick people who are correctly identified as sick.
 2. True Negative: Healthy people who are correctly identified as healthy.
 3. False Positive: Healthy people identified as sick.
 4. False Negative: Sick peope identified as healthy.
- True positive = correctly identified
- False positive = incorrectly identified
- True negative = correctly rejected
- False negative = incorrectly rejected

Hence, Precision (%Totally identified): correctly_identified / total_identified

and, 
Recall (%Validity of correctly identified): correctly_identified / (correctly_identified + incorrectly rejected)

Also, F1 (Harmonic mean of precision and recall) = 2 * (precision * recall) / (precision + recall)

The harmonic mean is particularly sensitive to a single lower-than average value. 
 

Results,
- Accuracy: 0.988513998564
- Precision: 0.972067039106
- Recall: 0.940540540541
- F1 Score: 0.956043956044

Predicts "Congratulations! You have won 10000000 dollars, hurrah!" as Ham, 
"Hey, how are you?" as Spam.
