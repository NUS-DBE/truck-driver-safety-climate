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
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')
pd.set_option("display.max_columns",100)
# from autogluon.tabular import TabularDataset, TabularPredictor
# import shap
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

df = pd.read_csv('trucking factors only (wo TSF).csv')
X = df.copy()[['OSC1', 'OSC2', 'OSC3', 'GSC1', 'GSC2', 'GSC3']]
YY=df.copy()[['acc']].to_numpy()
YY=YY.reshape(len(YY),)




def count(target,predict):
    b=np.unique(predict)
    for bb in b:   #0 1 2
        index=np.array(np.where(predict == bb))
        c=target[index]
        c=c.reshape(len(c[0]),)
        print('count')
        sc=sum(c)
        print(sc,len(c)-sc)

# def elbow_plot(X):
#     # find the optimal number of clusters using elbow method
#     WCSS = []
#     for i in range(1, 22):
#         model = KMeans(n_clusters=i, init='k-means++', random_state=0)
#         model.fit(X)
#         WCSS.append(model.inertia_)
#
#     fig = plt.figure(figsize=(10, 10))
#     plt.plot(range(1, 22), WCSS, linewidth=2, markersize=4, marker='o', color='green')
#     plt.xticks(np.arange(22))
#     plt.title("Elbow method")
#     plt.xlabel("Number of clusters")
#     plt.ylabel("WCSS")
#     plt.show()
#
#
# elbow_plot(X)

model = KMeans(n_clusters = 3, init = 'k-means++', random_state=0)
model.fit(X)
y_kmeans = model.predict(X)
#count = np.count_nonzero(y_kmeans)
count0 = np.count_nonzero(y_kmeans==0)
count1 = np.count_nonzero(y_kmeans==1)
# count2 = np.count_nonzero(y_kmeans==2)
print("0 = " + str(count0))
print("1 = " + str(count1))
# print("2 = " + str(count2))


# m = DBSCAN(eps=0.946, min_samples=230)
# m.fit(X)
# labels = m.labels_
# print(labels)



# import numpy as np
#
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets import make_blobs
# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
#
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
#
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
#
# print("number of estimated clusters : %d" % n_clusters_)



import time as time

from sklearn.cluster import AgglomerativeClustering

print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 3  # number of regions
ward = AgglomerativeClustering(
    n_clusters=n_clusters, linkage="ward")
ward.fit(X)
labels = ward.labels_
# # print(f"Elapsed time: {time.time() - st:.3f}s")
# # print(f"Number of pixels: {label.size}")
# # print(f"Number of clusters: {np.unique(label).size}")

y_kmeans = labels
count(YY,y_kmeans)
# y_kmeans=labels
FP=0
FN=0
TP=0
TN=0
acc=0
err=0
print(sum(YY))
for i in range(len(YY)):
    if YY[i]==0 and y_kmeans[i]==0:
        TN=TN+1
    elif YY[i]==0 and y_kmeans[i]==1:
        FP=FP+1
    elif YY[i] == 1 and y_kmeans[i] == 0:
        FN = FN+1
    else:
        TP=TP+1

print(TN,FP,FN,TP)
# print(acc/len(YY),err/len(YY))


# my_rho = np.corrcoef(YY, y_kmeans)

# print(my_rho)