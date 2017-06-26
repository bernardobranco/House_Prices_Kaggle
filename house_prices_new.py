import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

# substituting categorical variables with dummy variables
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

#alphas = np.linspace(0.0001, 100, 200)
alphas = [1, 0.1, 0.001, 0.0005]
alphas = np.linspace(0.0005,1,200)
model_lasso = linear_model.LassoCV(alphas=alphas,max_iter=5000).fit(X_train, y)
print(rmse_cv(model_lasso).mean())
#print(model_lasso.get_params())

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()

# Postive 10 coefficients (head)
print(coef.sort_values().head(10))
# Negative 10 coefficients (tail)
print(coef.sort_values().tail(10))

coef_neg = ['RoofMatl_ClyTile','MSZoning_C (all)','Condition2_PosN','Neighborhood_Edwards','SaleCondition_Abnorml','MSZoning_RM','CentralAir_N','GarageCond_Fa','LandContour_Bnk','SaleType_WD']
coef_pos = ['OverallQual','KitchenQual_Ex','Exterior1st_BrkFace','Neighborhood_NridgHt','LotArea','Functional_Typ','Neighborhood_NoRidge','Neighborhood_Crawfor','Neighborhood_StoneBr','GrLivArea']

#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
print(preds)
outliers = preds[abs(preds["residuals"]) > 0.2]
print(outliers)
plt.show()



