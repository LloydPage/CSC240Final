import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.manifold import TSNE
import pandas
from collections import Counter

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

data = pandas.read_csv("Cluster_Set.csv")
dataWhole = data
data = data.drop(["Name"], axis=1)
truth = Counter()
for x in data["Development Status"]:
    truth[x] +=1
print(truth)

data = data.drop(["Development Status"], axis=1)

#HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None), metric='euclidean', min_cluster_size=5, min_samples=None, p=None)


#Transform 5d to 2d for PLOT CHART ONLY - does not effect clustering
projection = TSNE().fit_transform(data)
plt.scatter(*projection.T, **plot_kwds)

#Heiraricahl Density Based Scan
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=False).fit(data)

#Assign Colors
color_palette = sns.color_palette('muted', 8)
cluster_colors = [ color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
#Weighted Colors = uncomment if wanted
#cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)


correct = ['' for x in range(len(data))]
correctCounter = Counter()
cnt = Counter()
i = 0
for x in clusterer.labels_:
    cnt[x] += 1
    if(x == 2 and dataWhole["Development Status"][i] == 0):
        correct[i] = "Correctly Clustered"
        correctCounter["Correct"] += 1
    elif(x == 1 and dataWhole["Development Status"][i] == 1):
        correct[i] = "Correctly Clustered"
        correctCounter["Correct Dev"] += 1
    elif(x == 0 and dataWhole["Development Status"][i] == 1):
        correct[i] = "Correctly Clustered"
        correctCounter["Correct Dev"] += 1
    elif(x == -1):
        correct[i] = "Failed to Clustered"
        correctCounter["Failed"] += 1
    else:
        correct[i] = "Incorrectly Clustered"
        correctCounter["Incorrect"] += 1
    i += 1
print (cnt)
print (correctCounter)


dataWhole["Cluster"] = clusterer.labels_
dataWhole["Correctness"] = correct




dataWhole.to_csv('Cluster_Set_Results.csv', index=False)
plt.show()


