import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

coef_neg = ['RoofMatl_ClyTile','MSZoning_C (all)','Condition2_PosN','Neighborhood_Edwards','SaleCondition_Abnorml','MSZoning_RM','CentralAir_N','GarageCond_Fa','LandContour_Bnk','SaleType_WD']
coef_pos = ['OverallQual','KitchenQual_Ex','Exterior1st_BrkFace','Neighborhood_NridgHt','LotArea','Functional_Typ','Neighborhood_NoRidge','Neighborhood_Crawfor','Neighborhood_StoneBr','GrLivArea']

features = ['RoofMatl_ClyTile','MSZoning_C (all)','Condition2_PosN','Neighborhood_Edwards','SaleCondition_Abnorml','MSZoning_RM','CentralAir_N','GarageCond_Fa','LandContour_Bnk','SaleType_WD','OverallQual','KitchenQual_Ex','Exterior1st_BrkFace','Neighborhood_NridgHt','LotArea','Functional_Typ','Neighborhood_NoRidge','Neighborhood_Crawfor','Neighborhood_StoneBr','GrLivArea']
#features = coef_pos

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters).fit(X_train[features])
print(kmeans.score(X_train[features]))
clusters_pred = kmeans.predict(X_train[features])

def getClusters(clusters_pred,x,y):
    clusters = []
    y_clusters = []
    for elem in range(num_clusters):
        cluster = pd.DataFrame()
        #y_cluster = pd.DataFrame()
        y_cluster = []
        for index in range(X_train.shape[0]):
            if clusters_pred[index] == elem:
                cluster = pd.concat((cluster,x.loc[index]),axis=1)
                y_cluster.append(y[index])
        clusters.append(cluster)
        y_clusters.append(y_cluster)
    clusters_transposed = [elem.transpose() for elem in clusters]
    return clusters_transposed,y_clusters

# Model
def rmse_cv(model,x,y):
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse

def lassoModel(x,y):
    alphas = np.linspace(0.0005,1,200)
    #alphas = [1, 0.1, 0.001, 0.0005]
    model_lasso = linear_model.LassoCV(alphas=alphas,max_iter=100000).fit(x, y)
    print(rmse_cv(model_lasso,x,y).mean())
    return model_lasso

def train_crossVal(feat_clusters,y_clusters):
    models = []
    for i in range(len(y_clusters)):
        models.append(lassoModel(feat_clusters[i],y_clusters[i]))
    return models


def getScores(models,feat_clusters,y_clusters):
    rmse_scores = []
    rmse_means = []
    num_houses = X_train.shape[0]
    for i in range(len(y_clusters)):
        rmse_score = rmse_cv(models[i],feat_clusters[i],y_clusters[i]).mean()
        rmse_scores.append(rmse_score)
        rmse_means.append(rmse_score * (len(y_clusters[i]) / num_houses))
    return np.sum(rmse_means)

feat_clusters,y_clusters = getClusters(clusters_pred,X_train,y)
models = train_crossVal(feat_clusters,y_clusters)
print("Trials:")
print(getScores(models,feat_clusters,y_clusters))


def getResiduals(models,feat_clusters,y_clusters):
    for i in range(len(y_clusters)):
        matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
        preds = pd.DataFrame({"preds":models[i].predict(feat_clusters[i]), "true":y_clusters[i]})
        preds["residuals"] = preds["true"] - preds["preds"]
        preds.plot(x = "preds", y = "residuals",kind = "scatter")
        outliers = preds[abs(preds["residuals"]) > 0.2]
        print(outliers)
        plt.title("Residuals from cluster %s:" % (i))
        plt.show()

#getResiduals(models,feat_clusters,y_clusters)