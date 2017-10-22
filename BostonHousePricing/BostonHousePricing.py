from datasets.boston_housing_data import get_boston_data
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
import os


def stats(attribute):
    print 'Min price: ${:,.2f}'.format(np.min(attribute))
    print 'Max price: ${:,.2f}'.format(np.max(attribute))
    print 'Mean of price: ${:,.2f}'.format(np.mean(attribute))
    print 'Median of price: ${:,.2f}'.format(np.median(attribute))
    print 'Std deviation of price: ${:,.2f}'.format(np.std(attribute))


def performance_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)


def grid_search_fit_model_decision_tree_regressor(X, y):
    # Cross validation sets for GridSearchCV
    cv_sets = ShuffleSplit(test_size=0.2, random_state=1, n_splits=10, train_size=None)

    # Decision tree regressor
    regressor = DecisionTreeRegressor()

    params = {'max_depth': range(1, 11)}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)

    return grid.best_estimator_


def grid_search_fit_model_polynomial_regressor(X, y):
    # Cross validation sets for GridSearchCV
    cv_sets = ShuffleSplit(test_size=0.2, random_state=1, n_splits=10, train_size=None)

    # Decision tree regressor
    regressor = DecisionTreeRegressor()

    params = {'max_depth': range(1, 11)}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    grid = grid.fit(X, y)

    return grid.best_estimator_


def graph_it(X, y, y_predict):
    plt.clf()
    plt.scatter(X, y)
    plt.scatter(X, y_predict, color="black")
    plt.xlabel('RM')
    plt.ylabel('prices')
    plt.legend(loc=2)
    plt.show()


if __name__ == "__main__":

    boston_housing_df = get_boston_data(os.path.abspath('datasets/boston_house_prices.csv'))

    prices = boston_housing_df['MEDV']

    # Prices in thousands
    prices = prices * 1000
    features = boston_housing_df.drop('MEDV', axis=1)

    stats(prices)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, prices, random_state=1, test_size=0.2)

    # Get the best regression model using grid search
    best_decision_tree_regressor_estimator = grid_search_fit_model_decision_tree_regressor(X_train, y_train)
    print "Parameter max_depth or the optimal model: {:,.2f}".format(best_decision_tree_regressor_estimator
                                                                     .get_params()['max_depth'])

    # Make predictions
    y_predict = best_decision_tree_regressor_estimator.predict(X_test)

    # How good are the predictions?
    print "R2_score for prediction: {:,.2f}".format(performance_metric(y_test, y_predict))

    graph_it(X_test['RM'], y_test, y_predict)
