from sklearn.datasets import load_boston
import pandas
import matplotlib.pyplot as plt
import math


def get_train_test_df(boston):
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
    train_data['target'] = train_target

    test_data = pandas.DataFrame(test_list, columns=boston.feature_names)
    test_data['target'] = test_target

    return [train_data, test_data]


def plot_histogram(data_frame):

    for column in data_frame.columns:
        attr = train_data[column]
        plt.hist(attr, bins=10)
        plt.show()


def pearson_coeff(data_frame):
    print data_frame.corr().ix[-1][:-1]


def standardize_train(data_frame):
    mean = data_frame.mean()
    std = data_frame.std()
    return [mean, std, ((data_frame - mean) / std)]


def standardize_test(data_frame, mean, std):
    return [mean, std, ((data_frame - mean) / std)]


def compute_h_theta(theta, row, train_data):
    h_theta_i = 0
    for i in range(0, len(train_data.columns) - 1):
        h_theta_i += theta[i] * row[train_data.columns[i]]
    return h_theta_i


def add_row(row, train_data):
    sum = 0
    for i in range(0, len(train_data.columns) - 1):
        sum += row[train_data.columns[i]]
    return sum


def sum_multiply_row(row, constant):
    sum = 0
    for i in range(0, len(train_data.columns) - 1):
        sum += row[i] * constant
    return sum


def modify_theta(theta, alpha, slope, train_data):
    for j in range(0, len(train_data.columns) - 1):
        theta[j] = theta[j] - alpha * slope
    return theta


def print_convergence(theta, train_data):

    sum = 0
    for index,row in train_data.iterrows():
            h_theta_i = compute_h_theta(theta, row, train_data)
            y_i = row['target']
            constant = math.pow((h_theta_i - y_i), 2)
            sum += constant

    slope = sum / (2 * len(train_data))
    return slope


if __name__ == "__main__":
    boston = load_boston()
    train_data, test_data = get_train_test_df(boston)

    plot_histogram(train_data)
    print pearson_coeff(train_data)

    mean, std, train_data = standardize_train(train_data)
    test_data = standardize_test(test_data, mean, std)

    theta = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    alpha = 0.01

    # Find the losses
    for i in range(0, 10000):
        losses = []
        squared_error = 0
        for index, row in train_data.iterrows():
            h_theta_i = compute_h_theta(theta, row, train_data)
            y_i = row['target']
            loss = (h_theta_i - y_i)
            losses.append(loss)

            # Just for printing purpose, calculate squared error
            constant = math.pow((h_theta_i - y_i), 2)
            squared_error += constant

        slope = squared_error / (2 * len(train_data))
        print slope

        train_data_matrix = train_data.as_matrix()
        train_data_matrix_transpose = train_data_matrix.transpose()

        for i in range(0, len(train_data_matrix_transpose) - 1):
            sum = 0
            for j in range(0, len(train_data_matrix_transpose[0])):
                sum += train_data_matrix_transpose[i][j] * losses[j]
            theta[i] = theta[i] - (alpha * sum) / 433

    print theta

