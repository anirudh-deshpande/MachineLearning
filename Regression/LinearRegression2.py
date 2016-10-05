from sklearn.datasets import load_boston
import pandas
import matplotlib.pyplot as plt
import numpy
import math
import sys
import itertools


def plot_histogram(data_frame):
    for column in data_frame.columns:
        attr = data_frame[column]
        plt.hist(attr, bins=10)
        plt.title(column)
        plt.show()


def pearson(column, target):
    sum_column = sum(column)
    sum_target = sum(target)
    squares_sum = sum([n * n for n in column])
    squares_target = sum([n * n for n in target])
    product_sum = 0
    for i in range(len(column)):
        product_sum += column[i]*target[i]
    size = len(column)
    numerator = product_sum - ((sum_column * sum_target) / size)
    denominator = math.sqrt((squares_sum - (sum_column * sum_column) / size) * (squares_target - (sum_target * sum_target) / size))
    if denominator == 0:
       return 0
    return float(numerator) / float(denominator)


def  standardize_train(data_frame):
    mean = data_frame.mean()
    std = data_frame.std()
    return [mean, std, ((data_frame - mean) / std)]


def standardize_train_columns(data_frame, data_frame_2, columns):
    for column in columns:
        mean = data_frame[column].mean()
        std = data_frame[column].std()
        data_frame[column] = (data_frame[column] - mean) / std
        data_frame_2[column] = (data_frame_2[column] - mean) / std


def standardize_target_train(target_frame):
    mean = target_frame.mean()
    std = target_frame.std()
    return [mean, std, ((target_frame - mean) / std)]


def standardize_target_test(mean, std, target_frame):
    return ((target_frame - mean) / std)


def standardize_test(data_frame, mean, std):
    return ((data_frame - mean) / std)


def add_x0(train_data, test_data):
    train_data.insert(0, 'x_0', 1)
    test_data.insert(0, 'x_0', 1)
    return [train_data, test_data]


def get_train_test_standardized_df(boston):
    test_list = []
    test_target = []
    train_list = []
    train_target = []

    for i in range(0, len(boston.data)):
        if i % 7 == 0:
            test_list.append(boston.data[i])
            test_target.append(boston.target[i])
        else:
            train_list.append(boston.data[i])
            train_target.append(boston.target[i])

    train_data = pandas.DataFrame(train_list, columns=boston.feature_names)
    test_data = pandas.DataFrame(test_list, columns=boston.feature_names)

    # plot_histogram(train_data)
    #
    # for column in train_data.columns:
    #     print column + ' :' + str(pearson(train_data[column], train_target))

    train_target = pandas.DataFrame(train_target, columns=['target'])
    test_target = pandas.DataFrame(test_target, columns=['target'])

    mean, std, train_data = standardize_train(train_data)
    test_data = standardize_test(test_data, mean, std)

    train_data, test_data = add_x0(train_data, test_data)

    return [train_data,train_target, test_data, test_target]


def pearson_coefficient(data_frame):
    print data_frame.corr().ix[-1][:-1]


def compute_pseudo_inverse_linear(data_frame):
    X = data_frame
    #print X.ix[1]
    X_transpose = X.transpose()
    #print X_transpose.ix[:, 0]
    dot_product = X_transpose.dot(X)
    #print dot_product
    inverse = numpy.linalg.pinv(dot_product)
    pseudo_inverse = inverse.dot(X_transpose)
    return pseudo_inverse


def compute_pseudo_inverse_ridge(data_frame, lambda_value):
    X = data_frame
    X_transpose = X.transpose()
    dot_product = X_transpose.dot(X)
    I = numpy.identity(len(dot_product))

    lambda_I = numpy.multiply(lambda_value, I)

    lambda_I_df = pandas.DataFrame(lambda_I, columns=data_frame.columns)
    regularized = pandas.DataFrame(dot_product.values + lambda_I_df.values, columns=dot_product.columns)

    inverse = numpy.linalg.inv(regularized)
    pseudo_inverse = inverse.dot(X_transpose)
    return pseudo_inverse


def compute_theta(pseudo_inverse, target):
    return pseudo_inverse.dot(target)


def compute_result(data_frame, theta):
    result = []

    for index, row in data_frame.iterrows():

        i = 0
        sum = 0
        for val in row:
            sum += val * theta[i]
            i += 1
        result.append(sum)
    return result


def compute_mse(target, results):
    target = target.as_matrix()
    N = len(target)
    loss = 0.0

    for i in range(0,N):
        loss += math.pow((target[i][0] - results[i]), 2)

    return loss / N


def linear_regression(train_data, train_target, test_data, test_target):
    pseudo_inverse = compute_pseudo_inverse_linear(train_data)
    theta = compute_theta(pseudo_inverse, train_target)

    result_test = compute_result(test_data, theta)
    mse_test = compute_mse(test_target, result_test)

    result_train = compute_result(train_data, theta)
    mse_train = compute_mse(train_target, result_train)

    return [result_train,result_test, mse_train, mse_test]


def ridge_regression(train_data, train_target, test_data, test_target, lambda_value):
    pseudo_inverse = compute_pseudo_inverse_ridge(train_data, lambda_value)

    theta = compute_theta(pseudo_inverse, train_target)
    #print theta

    result_test = compute_result(test_data, theta)
    mse_test = compute_mse(test_target, result_test)

    result_train = compute_result(train_data, theta)
    mse_train = compute_mse(train_target, result_train)

    return [mse_train, mse_test]
    # print 'Test result lambda = ' + str(lambda_value) + ' : ' + str(mse_test)
    # print 'Train result lambda = ' + str(lambda_value) + ' : ' + str(mse_train) + '\n'


def cross_validation(train_data):
    lambda_array = numpy.arange(0.0001, 10, 0.01)
    best_lambda = 10
    min_average = sys.float_info.max

    k_folds = []
    k_fold_target = []
    i = 0
    iteration = 1
    while i < len(train_data):
        start = i
        if iteration in [1, 2, 3]:
            end = i + 44
        else:
            end = i + 43
        k_folds.append(train_data[start:end])
        k_fold_target.append(train_target[start: end])

    for lambda_value in lambda_array:
        sum = 0
        i = 0
        iteration = 1
        while iteration <=10:
            cv_test = k_folds[iteration]
            cv_target_test = k_fold_target[iteration]
            cv_train = pandas.concat(k_folds[:iteration]+k_folds[iteration+1:])
            cv_target_train = pandas.concat(k_fold_target[:iteration]+k_fold_target[iteration+1:])
            result = ridge_regression(cv_train, cv_target_train, cv_test, cv_target_test, lambda_value)
            sum += result[1]
            iteration += 1
        average = sum / 10.0
        if average < min_average:
            min_average = average
            best_lambda = lambda_value
    return [float(best_lambda), float(min_average)]


def feature_selection_for_4_features(train_data, test_data):
    list_pearson = []
    for column in train_data.columns:
        tuple_value = (column, abs(float(pearson(train_data[column], train_target.as_matrix()))))
        list_pearson.append(tuple_value)

    list_top_4 = sorted(list_pearson, key=lambda x: x[1], reverse=True)[:4]
    list_column = ['x_0']
    for entry in list_top_4:
        list_column.append(entry[0])

    new_train_data = train_data[list_column]
    new_train_target = train_target
    new_test_target = test_target
    new_test_data = test_data[list_column]

    return linear_regression(new_train_data, new_train_target, new_test_data, new_test_target)


def get_list(train_data, train_target, which_list):
    if which_list == "top_4_pearson":
        list_pearson = []
        for column in train_data.columns:
            tuple_value = (column, abs(float(pearson(train_data[column], train_target.as_matrix()))))
            list_pearson.append(tuple_value)

        list_top_4 = sorted(list_pearson, key=lambda x: x[1], reverse=True)[:4]
        print 'Top 4 features' + str(list_top_4)
        list_column = ['x_0']
        for entry in list_top_4:
            list_column.append(entry[0])
        return [list_column]

    elif which_list == "brute_force":
        list_columns = list(train_data.columns)
        list_columns.remove('x_0')
        brute_force_list = [list(x) for x in itertools.combinations(list_columns, 4)]
        for index in xrange(0, len(brute_force_list)):
            entry = brute_force_list[index]
            entry = ['x_0'] + entry
            brute_force_list[index] = entry
        return brute_force_list


def feature_selection_for_4_features(train_data, test_data, which_list):
    train_test_mse_list = []
    iter_list = get_list(train_data, train_target,which_list)

    min_train = sys.float_info.max
    min_test = sys.float_info.max
    min_train_list = []
    min_test_list = []

    for list_column in iter_list:
        new_train_data = train_data[list_column]
        new_train_target = train_target
        new_test_target = test_target
        new_test_data = test_data[list_column]

        [result_train, result_test, mse_train, mse_test] = linear_regression(new_train_data, new_train_target, new_test_data, new_test_target)

        if mse_train < min_train:
            min_train = mse_train
            min_train_list = list_column

        if mse_test < min_test:
            min_test = mse_test
            min_test_list = list_column

        lr_result_tuple = ([mse_train, mse_test], list_column)
        train_test_mse_list.append(lr_result_tuple)

    return [train_test_mse_list, min_train, min_train_list, min_test, min_test_list]


def get_new_train_test_data(train_data, test_data):

    list_columns = list(train_data.columns)

    list_column_name = []
    length = len(train_data.columns)
    for i in xrange(1, length):
        for j in xrange(i, length):
            column_name = 'f_' + str(i) + str(j)
            list_column_name.append(column_name)
            train_data[column_name] = train_data[list_columns[i]] * train_data[list_columns[j]]
            test_data[column_name] = test_data[list_columns[i]] * test_data[list_columns[j]]

    standardize_train_columns(train_data, test_data, list_column_name)
    return [train_data, test_data]


def get_highest_correlated(train_data, train_target_list):
    list_columns = list(train_data.columns)
    max_correlation = sys.float_info.min
    max_correlation_feature = ''
    length = len(train_data.columns)
    for i in xrange(1, length):
        correlation = abs(pearson(train_data[list_columns[i]], train_target_list))
        if correlation > max_correlation:
            max_correlation = correlation
            max_correlation_feature = list_columns[i]
    return [max_correlation_feature, max_correlation]


def get_residue(list_a, list_b):
    difference = []
    for i in xrange(len(list_a)):
        difference.append(list_a[i] - list_b[i])
    return difference


def get_residue_based_mse(train_data, train_target, test_data, test_target):
    target = train_target.as_matrix()
    features = ['x_0']

    for i in xrange(0,4):
        highest_correlated = get_highest_correlated(train_data, target)
        features.append(highest_correlated[0])
        train = train_data[features]
        test = test_data[features]
        [result_train, result_test, mse_train, mse_test] = linear_regression(train, train_target,test,test_target)
        residue = get_residue(train_target.as_matrix(), result_train)
        target = residue
    return [mse_train, mse_test]


if __name__ == "__main__":
    boston = load_boston()
    train_data, train_target, test_data, test_target = get_train_test_standardized_df(boston)

    # print '*************************** linear regression ********************************** \n'
    # [result_train, result_test, mse_train, mse_test] = linear_regression(train_data, train_target, test_data, test_target)
    # print 'test result: ' + str(mse_test)
    # print 'train result: ' + str(mse_train) + '\n'
    #
    # print '*************************** ridge regression ********************************** \n'
    # lambda_array = [0.01, 0.1, 1.0]
    # for lambda_value in lambda_array:
    #     [mse_train, mse_test] = ridge_regression(train_data, train_target, test_data, test_target, lambda_value)
    #     print 'test result, lambda as ' + str(lambda_value) + ': ' + str(mse_test)
    #     print 'train result, lambda as ' + str(lambda_value) + ': ' + str(mse_train) + '\n'


    #cross validation
    print '*************************** cross validation ********************************** \n'

    result = cross_validation(train_data)
    print '\nLowest mse is %.4f for lambda %.2f '%(result[1], result[0])

    # #feature selection
    # print '\n *************************** feature selection ********************************** \n'
    #
    # [train_test_mse_list, min_train, min_train_list, min_test, min_test_list] = feature_selection_for_4_features(train_data, test_data, "top_4_pearson")
    #
    # for tuple_value in train_test_mse_list:
    #     print 'test result for top 4: ' + str(tuple_value[0][1])
    #     print 'train result for top 4: ' + str(tuple_value[0][0]) + '\n'
    #
    # [mse_train, mse_test] = get_residue_based_mse(train_data, train_target, test_data, test_target)
    # print 'Test result after residue based selection: ' + str(mse_test)
    # print 'Train result after residue based selection: ' + str(mse_train) + '\n'
    #
    #
    # [train_test_mse_list, min_train, min_train_list, min_test, min_test_list] = feature_selection_for_4_features(train_data, test_data, "brute_force")
    #
    # print 'minimum mse train brute force is '+ str(min_train) + ' for the features : ' + str(min_train_list[1:])
    # print 'minimum mse test brute force is ' + str(min_test) + ' for the features : ' + str(min_test_list[1:])
    #
    #
    # print '\n *************************** Polynomial feature expansion ********************************** \n'
    # [rest_train_data, rest_test_data] = get_new_train_test_data(train_data, test_data)
    # [result_train, result_test, mse_train, mse_test] = linear_regression(rest_train_data, train_target, rest_test_data, test_target)
    #
    # print 'Test result after ploynomial feature expansion: ' + str(mse_test)
    # print 'Train result after ploynomial feature expansion: ' + str(mse_train) + '\n'

