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
data = data.drop(["Name"], axis=1)
truth = Counter()
for x in data["Development Status"]:
    truth[x] +=1
print(truth)

#data = data.drop(["Development Status"], axis=1)

#HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None), metric='euclidean', min_cluster_size=5, min_samples=None, p=None)



projection = TSNE().fit_transform(data)
plt.scatter(*projection.T, **plot_kwds)

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=False).fit(data)
#print(clusterer.labels_)
#Assign Colors
color_palette = sns.color_palette('muted', 8)
cluster_colors = [ color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
#cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)

#clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep', 8))
cnt = Counter()
for x in clusterer.labels_:
    cnt[x] += 1
print (cnt)
plt.show()
