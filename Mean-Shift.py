
from rfpimp import *


import matplotlib.pyplot as plt



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.cluster import MeanShift, estimate_bandwidth

df = pd.read_csv('trucking factors only (wo TSF).csv')
X = df.copy()[['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']]
bandwidth = estimate_bandwidth(X, quantile=0.25)
m = MeanShift(bandwidth=bandwidth)
m.fit(X)

ms_labels = m.fit_predict(X)-1
print(sum(ms_labels))





#
# for i in [0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]:
#     bandwidth = estimate_bandwidth(X, quantile=i)
#     ms = MeanShift(bandwidth=bandwidth)
#     ms_labels = ms.fit_predict(X)
#     print(len(set(ms_labels)))
#     # Calculate the calinski_harabasz_score for MeanShift clustering
#     ch_score_ms = silhouette_score(X, ms_labels)
#
#     print(ch_score_ms)
#
# exit(0)


labels = m.labels_
Y=labels
unique_values, counts = np.unique(labels, return_counts=True)

from sklearn.model_selection import train_test_split  #数据分区

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=10, random_state=0).fit(X_train, y_train)

print(clf.score(X_test,y_test))

#
# features = [0, 1, (0, 1)]
# features = [1, 2, (1, 2)]
features=[0,1,2,3,4,5]



# from sklearn.inspection import permutation_importance
# feature_names=['OSC1','OSC2','OSC3','GSC1','GSC2','GSC3']
# scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
# r_multi = permutation_importance(
#      clf, X_test, y_test, n_repeats=30, random_state=0, scoring=scoring)
# for metric in r_multi:
#     print(f"{metric}")
#     r = r_multi[metric]
#     for i in r.importances_mean.argsort()[::-1]:
#
#         print(f"    {feature_names[i]:<8}"
#                 f"{r.importances_mean[i]:.3f}"
#                 f" +/- {r.importances_std[i]:.3f}")




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
    print(max(datadistri))
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
