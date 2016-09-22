import pandas
import scipy.stats


def get_csv(txt_file):
    csv_file = pandas.read_csv(txt_file, header=None)
    csv_file.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    return csv_file


def get_label_set(train_csv):
    label_set = set()
    for value in train_csv['K']:
        label_set.add(value)
    return label_set


def get_dict_mean_std_prior(label_set, train_csv):

    total_rows = len(train_csv)
    dict_mean = {}
    dict_std = {}
    dict_prior = {}

    iter_columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    for label in label_set:

        label_rows = train_csv.loc[train_csv['K'] == label]
        prior = len(label_rows) / float(total_rows)
        dict_prior[label] = prior

        for column in iter_columns:
            label_row_mean = label_rows[column].mean()
            label_row_std = label_rows[column].std()

            tuple_mean = (label, column)
            dict_mean[tuple_mean] = label_row_mean

            tuple_std = (label, column)
            dict_std[tuple_std] = label_row_std

    return [dict_mean, dict_std, dict_prior]


def get_prediction_accuracy(test_csv, dict_mean, dict_std, dict_prior, label_set):
    iter_columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    count = 0
    for index, row in test_csv.iterrows():
        max = 0
        prediction = 0

        for label in label_set:
            prior_of_label = dict_prior[label]

            gauss = prior_of_label


            for column in iter_columns:
                mean_of_label = dict_mean[(label, column)]
                std_of_label = dict_std[(label, column)]

                if mean_of_label != 0 or std_of_label != 0 or row[column] != 0:
                    gauss *= scipy.stats.norm(mean_of_label, std_of_label).pdf(row[column])

            if gauss > max:
                max = gauss
                prediction = label

        if prediction == row['K']:
            count+= 1

    return float(count)/len(test_csv)


def get_variance_zero(train_csv):
    label_set = get_label_set(train_csv)
    iter_columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    for label in label_set:
        label_rows = train_csv.loc[train_csv['K'] == label]

        for column in iter_columns:
            if label_rows[column].var() == 0:
                print label, column


def print_naivebayes_accuracy(train_file, test_file):
    train_csv = get_csv(train_file)
    label_set = get_label_set(train_csv)

    dict_mean, dict_std, dict_prior = get_dict_mean_std_prior(label_set, train_csv)
    test_csv = get_csv(test_file)

    print "*** Naive Bayes Results ***"
    print "----------------------------------------------------------------------------"

    train_accuracy = get_prediction_accuracy(train_csv, dict_mean, dict_std, dict_prior, label_set)
    print "Training accuracy : %.4f" % (train_accuracy * 100.0)

    test_accuracy = get_prediction_accuracy(test_csv, dict_mean, dict_std, dict_prior, label_set)
    print "Testing accuracy : %.4f" % (test_accuracy * 100.0)

    print '\n\n\n'

