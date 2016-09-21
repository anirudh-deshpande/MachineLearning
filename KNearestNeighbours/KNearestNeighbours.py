#Normalize the inputs
import pandas
from scipy.spatial import distance
import collections
from collections import Counter

def get_csv(txt_file):
    csv_file = pandas.read_csv(txt_file, header=None)
    csv_file.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    return csv_file


def get_label_set(csv_file):
    label_set = set()
    for val in csv_file['K']:
        label_set.add(val)
    return label_set


def normalize(csv_file_row, mean, std):
    csv_file_row = (csv_file_row - mean) / float(std)
    return csv_file_row


def normalize_train(train_file):

    dict_mean = {}
    dict_std = {}

    column_iter = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    for column in column_iter:
        mean_of_column = train_file[column].mean()
        dict_mean[column] = mean_of_column

        std_of_column = train_file[column].std()
        dict_std[column] = std_of_column

        train_file[column] = normalize(train_file[column], mean_of_column, std_of_column)

    return [train_file, dict_mean, dict_std]


def normalize_test(test_file, dict_mean, dict_std):
    column_iter = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    for column in column_iter:
        test_file[column] = normalize(test_file[column], dict_mean[column], dict_std[column])

    return test_file



def most_common(lst):
    data = Counter(lst)


    if data.most_common(1)[0][1] == len(lst):
        return data.most_common(1)[0][0]

    if data.most_common(1)[0][1] == 1:
        return None

    # Compare first 2 values. If same, the one which first occurs
    if data.most_common(2)[0][1] == data.most_common(2)[1][1]:
        for value in lst:
            if value == data.most_common(2)[0][0]:
                return value
            if value == data.most_common(2)[1][0]:
                return value

    return data.most_common(1)[0][0]

# def most_common(lst):
#     data = Counter(lst)
#
#     # Compare first 2 values. If same, return None
#     if data.most_common(1)[0][1] == len(lst):
#         return data.most_common(1)[0][0]
#
#     if data.most_common(2)[0][1] == data.most_common(2)[1][1]:
#         return None
#
#     if data.most_common(1)[0][1] == 1:
#         return None
#
#     return data.most_common(1)[0][0]



def get_dicts(train_csv, test_csv, training):
    correct_manhatten_1 = 0
    correct_euclidean_1 = 0
    correct_manhatten_3 = 0
    correct_euclidean_3 = 0
    correct_euclidean_5 = 0
    correct_euclidean_7 = 0
    correct_manhatten_7 = 0
    correct_manhatten_5 = 0

    for index1, test_row in test_csv.iterrows():
        # current_row is test_row accessed like row['B']
        current_test_row = tuple(test_row[1:-1])

        dict_euclidean = {}
        dict_manhatten = {}

        for index2, train_row in train_csv.iterrows():

            # each row is train_row accessed like row['B']
            current_train_row = tuple(train_row[1:-1])

            if training:
                if index1 == index2:
                    continue

            dst_euclidean = distance.euclidean(current_train_row, current_test_row)
            dict_euclidean[dst_euclidean] = int(train_row['K'])

            dst_manhatten = distance.cityblock(current_train_row, current_test_row)
            dict_manhatten[dst_manhatten] = int(train_row['K'])

        ordered_dict_euclidean = collections.OrderedDict(sorted(dict_euclidean.items()))
        ordered_dict_manhatten = collections.OrderedDict(sorted(dict_manhatten.items()))

        # For k = 1
        prediction = ordered_dict_manhatten[ordered_dict_manhatten.keys()[0]]
        if prediction == test_row['K']:
            correct_manhatten_1 += 1


        prediction = ordered_dict_euclidean[ordered_dict_euclidean.keys()[0]]
        if prediction == test_row['K']:
            correct_euclidean_1 += 1


        #For k = 3
        eu_first_three_values = ordered_dict_euclidean.values()[:3]
        prediction = most_common(eu_first_three_values)
        if prediction == None:
            prediction = eu_first_three_values[0]

        if prediction == test_row['K']:
            correct_euclidean_3 += 1


        mn_first_three_values = ordered_dict_manhatten.values()[:3]
        prediction = most_common(mn_first_three_values)
        if prediction == None:
            prediction = mn_first_three_values[0]
        if prediction == test_row['K']:
            correct_manhatten_3 += 1

        #For k = 5
        eu_first_three_values = ordered_dict_euclidean.values()[:5]
        prediction = most_common(eu_first_three_values)
        if prediction == None:
            prediction = eu_first_three_values[0]
        if prediction == test_row['K']:
            correct_euclidean_5 += 1

        mn_first_three_values = ordered_dict_manhatten.values()[:5]
        prediction = most_common(mn_first_three_values)
        if prediction == None:
            prediction = mn_first_three_values[0]

        if prediction == test_row['K']:
            correct_manhatten_5 += 1

        #For k = 7
        eu_first_three_values = ordered_dict_euclidean.values()[:7]
        prediction = most_common(eu_first_three_values)
        if prediction == None:
            prediction = eu_first_three_values[0]
        if prediction == test_row['K']:
            correct_euclidean_7 += 1

        mn_first_three_values = ordered_dict_manhatten.values()[:7]
        prediction = most_common(mn_first_three_values)
        if prediction == None:
            prediction = mn_first_three_values[0]
        if prediction == test_row['K']:
            correct_manhatten_7 += 1

    print "Manhatten Accuracy(L1) for K = 1 : %.4f" % (float(correct_manhatten_1) * 100.0/ len(test_csv))
    print "Eucledian Accuracy(L2) for K = 1 : %.4f" % (float(correct_euclidean_1) * 100.0/ len(test_csv))

    print "\nManhatten Accuracy(L1) for K = 3 : %.4f" % (float(correct_manhatten_3) * 100.0/ len(test_csv))
    print "Eucledian Accuracy(L2) for K = 3 : %.4f" % (float(correct_euclidean_3) * 100.0/ len(test_csv))

    print "\nManhatten Accuracy(L1) for K = 5 : %.4f" % (float(correct_manhatten_5) * 100.0/ len(test_csv))
    print "Eucledian Accuracy(L2) for K = 5 : %.4f" % (float(correct_euclidean_5) * 100.0/ len(test_csv))

    print "\nManhatten Accuracy(L1) for K = 7 : %.4f" % (float(correct_manhatten_7) * 100.0/ len(test_csv))
    print "Eucledian Accuracy(L2) for K = 7 : %.4f" % (float(correct_euclidean_7) * 100.0/ len(test_csv))


if __name__ == "__main__":

    train_csv = get_csv('train.txt')
    test_csv = get_csv('test.txt')

    train_csv, dict_mean, dict_std = normalize_train(train_csv)

    # dict_mean, dict_std are mean and std of
    test_csv = normalize_test(test_csv, dict_mean, dict_std)

    print "\nTraining accuracy -"
    print "-----------------------------------"
    get_dicts(train_csv, train_csv, training=True)

    print "\nTesting accuracy -"
    print "-----------------------------------"
    get_dicts(train_csv, test_csv, training=False)