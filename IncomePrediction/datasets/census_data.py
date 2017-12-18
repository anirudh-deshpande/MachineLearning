import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


def make_int(text):
    return int(text.strip())


def get_census_data(file_name):
    df = pd.read_csv(file_name,
                     index_col=False,
                     converters={
                            'age': make_int,
                            'workclass': strip,
                            'fnlwgt': make_int,
                            'education_level': strip,
                            'education-num':make_int,
                            'marital-status': strip,
                            'occupation': strip,
                            'relationship': strip,
                            'race': strip,
                            'sex': strip,
                            'capital - gain': make_int,
                            'capital-loss': make_int,
                            'hours-per-week': make_int,
                            'native-country': strip,
                            'income': strip
                         }
                     )
    return df


def plot_data_distribution(data, features):
    # Create figure
    fig = pl.figure(figsize=(11, 5))

    # Skewed feature plotting
    for i, feature in enumerate(features):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.hist(data[feature], bins=30, color='#00B000', edgecolor='black', linewidth=0.5)
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    fig.suptitle("Distributions of Continuous Census Data Features", \
                     fontsize=16, y=1.03)

    fig.tight_layout()
    fig.show()


def plot_correlationto_output(df, feature):
    fig = pl.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1)

    df_1 = pd.DataFrame(df.loc[df['income'] == '>50K'])
    df_2 = pd.DataFrame(df.loc[df['income'] == '<=50K'])

    ax.hist(df_1[feature], bins=30, color='#00B000',  alpha=0.5, edgecolor='black', linewidth=0.5)
    ax.hist(df_2[feature], bins=30, color='#00CCCC',  alpha=0.5, edgecolor='black', linewidth=0.5)

    ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Correlation")
    ax.set_ylim((0, 2000))
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    fig.tight_layout()
    fig.show()


def stats(census_data):
    total_records = len(census_data)
    greater_50k = len(census_data.loc[census_data['income'] == '>50K'])
    atmost_50k = len(census_data.loc[census_data['income'] == '<=50K'])
    greater_percent = 1.0 * greater_50k / total_records * 100
    atmost_percent = 1.0 * atmost_50k / total_records * 100

    print "Total records {}".format(total_records)
    print "Length greater than 50K {}".format(greater_50k)
    print "Length atmost 50K {}".format(atmost_50k)
    print "Percentage greater than 50K {}".format(greater_percent)
    print "Percentage atmost 50K {}".format(atmost_percent)


def apply_log_transformation(df, features):
    log_transformed = pd.DataFrame(data=df)
    log_transformed[features] = df[features].apply(lambda x: np.log(x + 1))
    return log_transformed


def apply_min_max_scaler(df, features):
    scaler = MinMaxScaler()
    minmax_scaled = pd.DataFrame(data=df)
    minmax_scaled[features] = scaler.fit_transform(df[features])
    return minmax_scaled


def get_census_data_df(file_name):
    adult_df = get_census_data(file_name)

    # select = adult_df.apply(lambda r: any([isinstance(e, basestring) for e in r]), axis=1)
    # adult_df = adult_df[~select]

    # select = adult_df[adult_df.apply(lambda row: row.astype(unicode).str.contains('\?', case=False).any(), axis=1)]
    # adult_df = adult_df.drop(select, axis=1)

    # Awesome way of dropping all rows with question mark
    adult_df = adult_df.replace({'?': np.nan}).dropna()

    print adult_df.shape

    income_raw = adult_df['income']
    features_raw = adult_df.drop('income', axis=1)

    # As 'capital-gain' & 'capital-loss' has most of the values at zero
    # =>
    # Most values tend to fall near zero.
    # Using a logarithmic transformation significantly reduces the range of values caused by outliers.

    # stats(adult_df)
    # to_plot = ['capital-loss', 'capital-gain', 'fnlwgt']
    # plot_data_distribution(adult_df, to_plot)

    skewed = ['capital-gain', 'capital-loss']
    features_raw_log_transformed = apply_log_transformation(features_raw, skewed)
    # plot_data_distribution(adult_df_log_transformed)

    # Numerical feature transformation
    numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_log_transformed_scaled = apply_min_max_scaler(features_raw_log_transformed, numerical)

    # Correlate the features
    # plot_correlationto_output(adult_df, 'fnlwgt')

    # One-hot-encode the categorical variables
    categorical = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    features_log_transformed_scaled_one_hot_encoded = pd.get_dummies(features_log_transformed_scaled, columns=categorical)

    # Final features
    features_final = features_log_transformed_scaled_one_hot_encoded

    # Transform income to binary
    income_final = income_raw.replace(['<=50K'], 0).replace(['>50K'], 1)

    # encoded = list(features_final.columns)
    # print encoded
    # print 'Total features after encoding {}'.format(len(encoded))

    return features_final, income_final

