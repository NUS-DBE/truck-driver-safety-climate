import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import openpyxl

from scipy import stats

from pandas.plotting import scatter_matrix
from pylab import rcParams
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from rfpimp import *
import imblearn
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
# %matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay


df = pd.read_csv('trucking factors only (wo TSF).csv')
X = df.copy()[['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']]
m = KMeans(n_clusters = 2, init = 'k-means++', random_state=0)
m.fit(X)
# X=X.to_numpy()
labels = m.labels_-1
Y=labels


from sklearn.model_selection import train_test_split  #数据分区

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)



# X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=10, random_state=0).fit(X_train, y_train)

print(clf.score(X_test,y_test))

#
# features = [0, 1, (0, 1)]
# features = [1, 2, (1, 2)]
features=[0,1,2,3,4,5]
#
pdp_display=PartialDependenceDisplay.from_estimator(clf, X_train, features,kind='both', centered=True)  #kind='both', centered=True
fig, ax = plt.subplots(figsize=(8, 6))
pdp_display.plot(ax=ax)
plt.title("Partial Dependence Plot")
plt.xlabel("Feature Values")
plt.ylabel("Partial Dependence")
plt.tight_layout()
plt.show()
# https://scikit-learn.org/stable/modules/partial_dependence.html

#
# import shap
# explainer = shap.Explainer(clf)
# shap_values = explainer(X_test)
# shap.plots.bar(shap_values, max_display=10)
#
# shap.plots.bar(shap_values.cohorts(2).abs.mean(0))
#
# shap.plots.heatmap(shap_values[1:100])
# # shap.plots.waterfall(shap_values[0]) # For the first observation
#
#
# shap.initjs()
# explainer = shap.TreeExplainer(clf)
#
#
# shap_values = explainer.shap_values(X_test)[1]
#
# shap.decision_plot(explainer.expected_value, shap_values, X_test)
