import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
def main():
    data=pandas.read_csv("Cluster_Set.csv")
    x=np.array(data.iloc[:,1:5])
    y=np.array(data.iloc[:,6])

    clf=DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4 )
    clf.fit(X_train, y_train)
    listdev=[]
    listndev=[]
    for r in range(len(y_test)):
        if y_test[r]==1:
            listdev.append(r)
        else:
            listndev.append(r)
    x_dev, y_dev=X_test[listdev], y_test[listdev]
    x_ndev, y_ndev=X_test[listndev],y_test[listndev]
    score=clf.score(X_test, y_test)
    scored=clf.score(x_dev, y_dev)
    scorex=clf.score(x_ndev, y_ndev)
    print(score)
    print(scored)
    print(scorex)
    fig=tree.plot_tree(clf)
    plt.show(fig)
if __name__=="__main__":
    main()

