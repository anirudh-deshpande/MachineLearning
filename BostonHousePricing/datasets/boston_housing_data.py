import pandas as pd

def get_boston_data(file_name):
    boston_housing_df = pd.read_csv(file_name)
    return boston_housing_df

# if __name__ == "__main__":
#     get_boston_data("boston_house_prices.csv")
