import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
def main():
    data=pandas.read_csv("Cluster_Set.csv")
    x=np.array(data.iloc[:,range(1,6)])
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
    #plotting due to Tree being to large to show normally
    n_classes = 2
    plot_colors = "rb"
    plot_step = .2
    for pairidx, pair in enumerate([[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]):
        x2=X_train[:,pair]
        x3=X_test[:,pair]
        x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1
        y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1
        #normalization to allow for running
        if pair[0]!=1 and pair[0]!=2 and pair[1]!=1 and  pair[1]!=2:
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            clf=DecisionTreeClassifier().fit(x2,y_train)
        elif pair[0]!=1 and pair[0]!=2:
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min/1000, y_max/1000, plot_step))
            x2[:,1]=x2[:,1]/1000
            x3[:,1]=x3[:,1]/1000
            clf=DecisionTreeClassifier().fit(x2,y_train)
        elif pair[1]!=1 and  pair[1]!=2:
            xx, yy = np.meshgrid(np.arange(x_min/1000, x_max/1000,plot_step),np.arange(y_min, y_max, plot_step))
            x2[:,0]=x2[:,0]/1000
            x3[:,0]=x3[:,0]/1000
            clf=DecisionTreeClassifier().fit(x2,y_train)
        else:
            xx, yy = np.meshgrid(np.arange(x_min/1000, x_max/1000, plot_step),np.arange(y_min/1000, y_max/1000, plot_step))            
            x2[:,[0,1]]=x2[:,[0,1]]/1000
            x3[:,[0,1]]=x3[:,[0,1]]/1000
            clf=DecisionTreeClassifier().fit(x2,y_train)
        print(x2[:,0].max(), x2[:,0].min(), x3[:,1].max(), x3[:,1].min())
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
        plt.xlabel(list(data.columns)[pair[0]+1])
        plt.ylabel(list(data.columns)[pair[1]+1])
            # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y_train == i)
            plt.scatter(x2[idx, 0], x2[idx, 1], c=color, label=y_train[i],cmap=plt.cm.RdBu, edgecolor='black', s=15)
            idx = np.where(y_test==i)
            plt.scatter(x3[idx, 0], x3[idx, 1], c=color, label=y_test[i],cmap=plt.cm.RdBu, edgecolor='black', s=15)

        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        #plt.figure()
        plt.show()




if __name__=="__main__":
    main()

