from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import openpyxl

from scipy import stats

from pandas.plotting import scatter_matrix
from pylab import rcParams
import sklearn


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
from sklearn.cluster import DBSCAN, Birch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay


df = pd.read_csv('trucking factors only (wo TSF).csv')
X = df.copy()[['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']]
m = Birch(n_clusters=2)
m.fit(X)
# X=X.to_numpy()
labels = m.labels_-1
Y=labels


#
# from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
# print(sum(m.labels_))
# labels = m.fit_predict(X)
#
# # 计算每个聚类的平均值作为近似聚类中心
# cluster_centers = []
# for cluster_label in range(2):
#     cluster_points = X[labels == cluster_label]
#     cluster_center = np.mean(cluster_points, axis=0)
#     cluster_centers.append(cluster_center)
#
# # 输出近似的聚类中心
# for i, center in enumerate(cluster_centers):
#     print(f"Cluster {i} center:", center)
# # Instantiate the clustering model and visualizer
# model = Birch()
# visualizer = KElbowVisualizer(model, k=(2,12), metric='calinski_harabasz', timings=False)
#
# visualizer.fit(X)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure
#
#
# visualizer = KElbowVisualizer(model, k=(2,12), metric='silhouette', timings=False)
#
# visualizer.fit(X)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure
#
# exit(0)





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


# from sklearn.inspection import permutation_importance
# feature_names=['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']
# scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
# r_multi = permutation_importance(
#      clf, X_test, y_test, n_repeats=30, random_state=0, scoring=scoring)
#
# for metric in r_multi:
#     print(f"{metric}")
#     r = r_multi[metric]
#     for i in r.importances_mean.argsort()[::-1]:
#         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#             print(f"    {feature_names[i]:<8}"
#                   f"{r.importances_mean[i]:.3f}"
#                   f" +/- {r.importances_std[i]:.3f}")
pdp_display=PartialDependenceDisplay.from_estimator(clf, X_train, features,kind='both', centered=True)  #kind='both', centered=True
fig, ax = plt.subplots(figsize=(8, 6))
plt.title("ICE and PDP representations")
pdp_display.plot(ax=ax)
# plt.title("Partial Dependence Plot")
# plt.xlabel("Feature Values")
plt.ylabel("Partial Dependence")
plt.tight_layout()
plt.show()
# exit(0)
# # https://scikit-learn.org/stable/modules/partial_dependence.html
#
#
#
from klcompution import distribution_value
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 12))

mse=0
for i in range(6):

    datadistri=pdp_display.pd_results[i].average[0]
    x_values,normalized_y_values,kl,kl_data=distribution_value(datadistri)
    ax=axes[i//3, i%3]

    print(np.sum(kl_data * np.log(kl_data / normalized_y_values)))#should >0
    print(np.trapz(normalized_y_values, x=x_values))
    print(np.trapz(kl_data, x=x_values))

    ax.plot(x_values, normalized_y_values, label='PDP')
    ax.plot(x_values, kl_data, label='step function')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    # ax.set_title(f'KL Comparison:'+str(round(kl,2)))
    ax.set_title(f'MSE Comparison:' + str(round(kl, 3)))
    ax.grid()


    print("Kullback-Leibler Divergence:", kl)
    mse=mse+kl
    # print(max(datadistri))
# 调整子图布局，避免重叠
plt.tight_layout()
plt.legend()
plt.show()
print('mse',mse)
# exit(0)

import shap
explainer = shap.Explainer(clf)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, max_display=10)

shap.plots.bar(shap_values.cohorts(2).abs.mean(0))

shap.plots.heatmap(shap_values[1:1000])
# shap.plots.waterfall(shap_values[0]) # For the first observation


shap.initjs()
explainer = shap.TreeExplainer(clf)


shap_values = explainer.shap_values(X_test)[1]

shap.decision_plot(explainer.expected_value, shap_values, X_test)
