import os
import math
import numpy as np
from datasets.census_data import get_census_data_df
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV



# f1_score inclined towards precision
def f1_score(precision, recall):
    beta_sq = math.pow(0.5, 2)
    f1_score = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return f1_score


def baseline_model():
    # Baseline: Always classified as >=50K
    TP = np.sum(income_final)  # No. correctly classified
    FP = income_final.count() - TP  # No. incorrectly classified
    TN = 0  # No. correct ones incorrectly classified
    FN = 0  # No. incorrect incorrectly classified

    precision = 1.0 * TP / (TP + FP)
    accuracy = precision
    recall = 1.0 * TP / (TP + FN)
    fscore = f1_score(precision, recall)

    print 'Accuracy: {}, Precision: {}, recall: {}, f1_score: {}'.format(accuracy, precision, recall, fscore)

    return accuracy, fscore


# Generic predictor
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

    results = {}

    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()

    results['train_time'] = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(predictions_train, y_train[:300])
    results['acc_test'] = accuracy_score(predictions_test, y_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    print '{} trained on {} samples.'.format(learner.__class__.__name__, sample_size)

    return results


def all_algorithms():
    clf_A = RandomForestClassifier()
    clf_B = SVC()
    clf_C = LogisticRegression()
    clf_D = DecisionTreeClassifier()
    clf_E = GaussianNB()
    clf_F = MultinomialNB()
    clf_G = BernoulliNB()
    clf_H = LinearSVC()
    clf_I = ExtraTreesClassifier()
    clf_J = MLPClassifier()
    clf_K = KNeighborsClassifier()

    clf_all = [clf_A, clf_B, clf_C, clf_D, clf_E, clf_F, clf_G, clf_H, clf_I, clf_J, clf_K]
    # clf_all = [clf_F, clf_G]

    return clf_all


def sample_sizes(X_train):

    # 100% of training set
    samples_100 = X_train.shape[0]

    # 10% of training set
    samples_10 = X_train.shape[0] / 10

    # 1% of training set
    samples_1 = X_train.shape[0] / 100

    return [samples_1, samples_10, samples_100]


def evaluate(results, accuracy, f1):
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(15, 10))

    # Constants
    bar_width = 0.05
    colors = ['#FF0000', '#97197A', '#33197A', '#5C7A36', '#E89B0C',
              '#470CE8', '#470C19', '#FF0D7B', '#0DFFCE', '#150C47',
              '#A936BE']

    # colors = ['#FF0000', '#97197A']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j / 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j / 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j / 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j / 3, j % 3].set_xlabel("Training Set Size")
                ax[j / 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictorsxcxc
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(handles=patches, bbox_to_anchor=(0.5, 0.05), \
              loc='upper right', borderaxespad=0., ncol=5, fontsize='small')

    # Aesthetics
    pl.suptitle("Performance Metrics for Twelve Supervised Learning Models", fontsize=16, y=1.10)
    pl.tight_layout()
    pl.show()


def grid_search(learner, param_grid, X_train, X_test, y_train, y_test):

    scorer = make_scorer(fbeta_score, beta=0.5)

    grid_obj = GridSearchCV(estimator=learner,
                            scoring=scorer,
                            param_grid=param_grid,
                            n_jobs=4,
                            verbose=1
                            )

    grid_fit = grid_obj.fit(X_train, y_train)

    best_clf = grid_fit.best_estimator_

    learner = learner.fit(X_train, y_train)
    predictions = learner.predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print "Unoptimized model\n------"
    print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
    print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5))
    print "\nOptimized Model\n------"
    print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
    print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5))


if __name__ == '__main__':
    features_final, income_final = get_census_data_df(os.path.abspath('datasets/adult-data.csv'))

    X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                        income_final,
                                                        test_size=0.2,
                                                        random_state=0)

    # Baseline model
    accuracy, fscore = baseline_model()

    # Initial model evaluation
    clf_all = all_algorithms()
    sample_sizes = sample_sizes(X_train)

    results = {}

    for clf in clf_all:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}

        for i, samples in enumerate(sample_sizes):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    print results

    evaluate(results, accuracy, fscore)

    # Multinomial Naive Bayes gives the best F-score
    # param_grid_1 = {
    #     'alpha': [0.01, .5, 1., 1.5, 2, 2.5, 5],
    #     'fit_prior': [True, False],
    #     'class_prior': [(i / 10, 1 - i / 10) for i in range(1, 10)],
    # }
    #
    # grid_search(MultinomialNB(), param_grid_1, X_train, X_test, y_train, y_test)
    #
    # param_grid_2 = {
    #     'alpha': [0.01, .5, 1., 1.5, 2, 2.5, 5],
    #     'binarize': [0, 1, 5, 10, 50, 100],
    #     'fit_prior': [True, False],
    #     'class_prior': [(i / 10, 1 - i / 10) for i in range(1, 10)],
    # }
    #
    # grid_search(BernoulliNB(), param_grid_2, X_train, X_test, y_train, y_test)

    param_grid_3 = {
        'n_estimators': [5, 10, 20, 50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 8, 10, 12],
        'min_samples_leaf': [2, 8, 10, 12],
        'min_impurity_decrease': [0., .1, .01, .001, .0001, .00001]
    }

    grid_search(RandomForestClassifier(), param_grid_3, X_train, X_test, y_train, y_test)

