## Income classification

Prediction of how much an individual makes annually.
Using UCI data-set: https://archive.ics.uci.edu/ml/datasets/Census+Income

## Features and target

    Target variable: >50K , <=50K.
     
    Features:
    1.  age: continuous. 
    2.  workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
    3.  fnlwgt: continuous. 
    4.  education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
    5.  education-num: continuous. 
    6.  marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
    7.  occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
    8.  relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
    9.  race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
    10. sex: Female, Male. 
    11. capital-gain: continuous. 
    12. capital-loss: continuous. 
    13. hours-per-week: continuous. 
    14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    
### Pre processing

#### Logarithmic transformation:
Is necessary when the data is skewed as the skewed features might negatively impact the learning algorithm. Applied logarithmic transformation on `capital-gain` and `capital-loss`.      <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a>
    
    log_transformed_data[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

    # As 'capital-gain' & 'capital-loss' has most of the values at zero
    # =>
    # Most values tend to fall near zero.
    # Using a logarithmic transformation significantly reduces the range of values caused by outliers.

#### Scaling
Is necessary for the numerical features. Otherwise the numerical features with higher magnitude might negatively influence the learning algorithm.
Eg.
 - <a href='http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler'>MaxAbsScaler</a>
 - <a href='http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler'>MinMaxScaler</a>
 - <a href='http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py'>All different scalers</a>
 
 
#### One hot encoding of categorical variables
A popular way of converting categorical variables to numerical variables

#### Precision or Recall important?

For classification problems that are skewed towards one outcome (75% of people make <=50K), accuracy is not the tight metric for classification purpose. 
 - Precision (Sensitivity): Among the right ones, how much you classified as right?
 - Recall (Specificity): How many you classified as right are actually right?
 
 We can assign a Beta score of 0.5 to emphasize more on precision.  
    
    Fβ=(1+β^2)⋅precision⋅recall(β^2⋅precision)+recall

