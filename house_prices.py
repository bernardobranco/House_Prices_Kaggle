import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

# Load train data set
df = pd.read_csv("./data/train.csv")

# removing variables with least number of entries and variables containing NANs entries
df = df.drop(["Alley","LotFrontage","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)
df = df.dropna()

# Converting categorical variables into dummy variables
dummies = pd.get_dummies(df[['MSZoning', 'Street',"LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","SaleType","SaleCondition"]])

# dropping categorical variables
X_ = df.drop(['SalePrice', 'MSZoning', 'Street', "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
              "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
              "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual",
              "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir",
              "Electrical", "KitchenQual", "Functional", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
              "PavedDrive", "SaleType", "SaleCondition"], axis=1).astype('float64')

# concatenating dummy variables to house features
X = pd.concat([X_, dummies[[
    'MSZoning_FV',
    'MSZoning_RH',
    'MSZoning_RL',
    'MSZoning_RM',

    'Street_Pave',

    'LotShape_IR2',
    'LotShape_IR3',
    'LotShape_Reg',

    'LandContour_HLS',
    'LandContour_Low',
    'LandContour_Lvl',

    'Utilities_NoSeWa',

    'LotConfig_CulDSac',
    'LotConfig_FR2',
    'LotConfig_FR3',
    'LotConfig_Inside',

    'LandSlope_Mod',
    'LandSlope_Sev',

    'Neighborhood_Blueste',
    'Neighborhood_BrDale',
    'Neighborhood_BrkSide',
    'Neighborhood_ClearCr',
    'Neighborhood_CollgCr',
    'Neighborhood_Crawfor',
    'Neighborhood_Edwards',
    'Neighborhood_Gilbert',
    'Neighborhood_IDOTRR',
    'Neighborhood_MeadowV',
    'Neighborhood_Mitchel',
    'Neighborhood_NAmes',
    'Neighborhood_NPkVill',
    'Neighborhood_NWAmes',
    'Neighborhood_NoRidge',
    'Neighborhood_NridgHt',
    'Neighborhood_OldTown',
    'Neighborhood_SWISU',
    'Neighborhood_Sawyer',
    'Neighborhood_SawyerW',
    'Neighborhood_Somerst',
    'Neighborhood_StoneBr',
    'Neighborhood_Timber',
    'Neighborhood_Veenker',

    'Condition1_Feedr',
    'Condition1_Norm',
    'Condition1_PosA',
    'Condition1_PosN',
    'Condition1_RRAe',
    'Condition1_RRAn',
    'Condition1_RRNe',
    'Condition1_RRNn',

    'Condition2_Feedr',
    'Condition2_Norm',
    'Condition2_PosA',
    'Condition2_PosN',
    'Condition2_RRAe',
    'Condition2_RRAn',
    'Condition2_RRNn',

    'BldgType_2fmCon',
    'BldgType_Duplex',
    'BldgType_Twnhs',
    'BldgType_TwnhsE',

    'HouseStyle_1.5Unf',
    'HouseStyle_1Story',
    'HouseStyle_2.5Fin',
    'HouseStyle_2.5Unf',
    'HouseStyle_2Story',
    'HouseStyle_SFoyer',
    'HouseStyle_SLvl',

    'RoofStyle_Gable',
    'RoofStyle_Gambrel',
    'RoofStyle_Hip',
    'RoofStyle_Mansard',
    'RoofStyle_Shed',

    'RoofMatl_CompShg',
    'RoofMatl_Membran',
    'RoofMatl_Metal',
    'RoofMatl_Roll',
    'RoofMatl_Tar&Grv',
    'RoofMatl_WdShake',
    'RoofMatl_WdShngl',

    'Exterior1st_BrkComm',
    'Exterior1st_BrkFace',
    'Exterior1st_CBlock',
    'Exterior1st_CemntBd',
    'Exterior1st_HdBoard',
    'Exterior1st_ImStucc',
    'Exterior1st_MetalSd',
    'Exterior1st_Plywood',
    'Exterior1st_Stone',
    'Exterior1st_Stucco',
    'Exterior1st_VinylSd',
    'Exterior1st_Wd Sdng',
    'Exterior1st_WdShing',

    'Exterior2nd_AsphShn',
    'Exterior2nd_Brk Cmn',
    'Exterior2nd_BrkFace',
    'Exterior2nd_CBlock',
    'Exterior2nd_CmentBd',
    'Exterior2nd_HdBoard',
    'Exterior2nd_ImStucc',
    'Exterior2nd_MetalSd',
    'Exterior2nd_Other',
    'Exterior2nd_Plywood',
    'Exterior2nd_Stone',
    'Exterior2nd_Stucco',
    'Exterior2nd_VinylSd',
    'Exterior2nd_Wd Sdng',
    'Exterior2nd_Wd Shng',

    'MasVnrType_BrkFace',
    'MasVnrType_None',
    'MasVnrType_Stone',

    'ExterQual_Fa',
    'ExterQual_Gd',
    'ExterQual_TA',
    'ExterCond_Ex',
    'ExterCond_Fa',
    'ExterCond_Gd',
    'ExterCond_TA',

    'Foundation_CBlock',
    'Foundation_PConc',
    'Foundation_Stone',
    'Foundation_Wood',

    'BsmtQual_Fa',
    'BsmtQual_Gd',
    'BsmtQual_TA',
    'BsmtCond_Fa',
    'BsmtCond_Gd',
    'BsmtCond_Po',
    'BsmtCond_TA',

    'BsmtExposure_Gd',
    'BsmtExposure_Mn',
    'BsmtExposure_No',
    'BsmtFinType1_ALQ',
    'BsmtFinType1_BLQ',
    'BsmtFinType1_GLQ',
    'BsmtFinType1_LwQ',
    'BsmtFinType1_Rec',
    'BsmtFinType1_Unf',
    'BsmtFinType2_ALQ',
    'BsmtFinType2_BLQ',
    'BsmtFinType2_GLQ',
    'BsmtFinType2_LwQ',
    'BsmtFinType2_Rec',
    'BsmtFinType2_Unf',

    'Heating_GasW',
    'Heating_Grav',
    'Heating_OthW',

    'HeatingQC_Fa',
    'HeatingQC_Gd',
    'HeatingQC_Po',
    'HeatingQC_TA',

    'CentralAir_Y',

    'Electrical_FuseF',
    'Electrical_FuseP',
    'Electrical_Mix',
    'Electrical_SBrkr',

    'KitchenQual_Fa',
    'KitchenQual_Gd',
    'KitchenQual_TA',

    'Functional_Maj2',
    'Functional_Min1',
    'Functional_Min2',
    'Functional_Mod',
    'Functional_Sev',
    'Functional_Typ',

    'GarageType_Attchd',
    'GarageType_Basment',
    'GarageType_BuiltIn',
    'GarageType_CarPort',
    'GarageType_Detchd',

    'GarageFinish_RFn',
    'GarageFinish_Unf',

    'GarageQual_Fa',
    'GarageQual_Gd',
    'GarageQual_Po',
    'GarageQual_TA',
    'GarageCond_Ex',
    'GarageCond_Fa',
    'GarageCond_Gd',
    'GarageCond_Po',
    'GarageCond_TA',

    'PavedDrive_P',
    'PavedDrive_Y',

    'SaleType_CWD',
    'SaleType_Con',
    'SaleType_ConLD',
    'SaleType_ConLI',
    'SaleType_ConLw',
    'SaleType_New',
    'SaleType_Oth',
    'SaleType_WD',

    'SaleCondition_AdjLand',
    'SaleCondition_Alloca',
    'SaleCondition_Family',
    'SaleCondition_Normal',
    'SaleCondition_Partial']]], axis=1)


# House prices: value we are trying to predict
y = df.SalePrice
'''
# Getting the log of the numeric values
y = np.log1p(y)

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
print(numeric_feats)
'''

# Ridge regression:

'''
def getFeatureVector(dim):
    i = 0
    feature_vec = []
    while i < dim:
        variable = input("Please enter a variable: ")
        feature_vec.append(X[variable])
        i += 1
    return feature_vec
'''
def getFeatureVector(all_vars):
    feature_vec = []
    if all_vars:
        variableNames = list(X.columns.values)[1:]
    else:
        variableNames = ["FullBath","GrLivArea","YearBuilt","GarageCars","OverallQual","YearRemodAdd","1stFlrSF","GarageArea","GarageYrBlt","MasVnrArea","KitchenQual_TA","ExterQual_TA","BsmtQual_TA","Foundation_PConc"]
    for elem in variableNames:
        feature_vec.append(X[elem])
    return feature_vec


# Cross-Validation

# loss function
def rmse_cv(model,feature_vec):
    rmse= np.sqrt(-cross_val_score(model, feature_vec, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
'''
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_lasso = [rmse_cv(linear_model.Lasso(alpha = alpha, max_iter=5000)).mean()
            for alpha in alphas]
print('Best score: '+cv_lasso.min())
'''
'''
model_lasso = linear_model.LassoCV(alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],max_iter=5000).fit(X, y)
score_lasso = rmse_cv(model_lasso,X).mean()
print(score_lasso)
'''

def crossValidation(model, featureVector):
    # print("Alpha: %0.4f" % (model.alpha))
    k_fold = KFold(n_splits=3)
    scores = cross_val_score(model, featureVector, y, cv=k_fold)
    # print("Accuracy: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print('---------------------------------')
    return (model.alpha, scores.mean())

'''
def trainRRModel():
    #dim = int(input("Input number of feature variables: "))
    feature_vector = getFeatureVector(False)
    features = pd.concat(feature_vector, axis=1)
    alphas = np.linspace(0.001, 100, 200)
    results = []
    for alpha in alphas:
        model = linear_model.Ridge(alpha)
        results.append(crossValidation(model, features))
    return results
'''
# Aux function to find best alpha parameter from cross validation
def findBest(res):
    highest_score = 0
    for i in range(0,len(res)):
        if res[i][1] > highest_score:
            highest_score = res[i][1]
            highest_alpha = res[i][0]
    print("Best model was with alpha %0.3f and mean score of %0.6f" % (highest_alpha,highest_score))



# Lasso - L1 regression
def trainRRL1Model():
    feature_vector = getFeatureVector(True)
    features = pd.concat(feature_vector, axis=1)
    alphas = np.linspace(0.001, 100, 200)
    results = []
    for alpha in alphas:
        model = linear_model.Lasso(alpha,max_iter=5000)
        results.append(crossValidation(model, X))
    return results

results = trainRRL1Model()
findBest(results)
#print(list(X.columns.values)[1:])
