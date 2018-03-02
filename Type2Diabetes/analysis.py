import first.forecast as ff
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def get_principal_component(data):
    estimator = PCA(n_components=2)
    data_pca = estimator.fit_transform(data)
    return data_pca


def plot_decision_boundary(pred_funcs, X, y):
    mins = np.min(X, axis=0) - .5
    maxs = np.max(X, axis=0) + .5
    stride = 0.01
    # Generate a grid of points with distance `stride` between them
    x1, x2 = np.meshgrid(np.arange(mins[0], maxs[0], stride), np.arange(mins[1], maxs[1], stride))
    g = gs.GridSpec(2, 2)
    titles = ['LogisticRegression', 'SVM', 'RandomForest', 'DecisionTree']
    plt.figure(num='Experiment', figsize=(9, 6))
    plt.suptitle('Boundary Analysis')
    for i in range(len(pred_funcs)):
        # Predict the function value for the whole grid, and as the height of contour
        z = pred_funcs[i](np.column_stack((x1.ravel(), x2.ravel()))).reshape(x1.shape)
        # Plot the contour and training examples
        plt.subplot(g[i//2, i % 2])
        plt.contour(x1, x2, z)
        colors = ['red', 'blue']
        markers = ['+', '_']
        for j in range(len(colors)):
            first_component = X[:, 0][y == j]
            second_component = X[:, 1][y == j]
            plt.scatter(first_component, second_component, color=colors[j], marker=markers[j], cmap=plt.cm.Spectral)
        plt.title(titles[i])
        plt.legend(['T2D', 'Others'], loc='lower right')
        # plt.xlabel('First Component')
        # plt.ylabel('Second Component')
    plt.savefig('Boundary Analysis.png')


def plot_pr_roc(pred_funcs, data, target):
    fpr = {}  # False Positive Rate
    tpr = {}  # True Positive Rate
    roc_auc = dict()  # Area Under the Curve (AUC)
    recall = {}
    precision = {}
    for i in range(len(pred_funcs)):
        y_score = pred_funcs[i](data)
        recall[i], precision[i], _ = precision_recall_curve(target, y_score)  # return 3rd is the threshold
        fpr[i], tpr[i], _ = roc_curve(target, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['green', 'blue', 'magenta', 'yellow']
    algorithms = ['LogisticRegression', 'SVM', 'RandomForest', 'DecisionTree']
    plt.close()
    plt.figure(num='Experiment', figsize=(10, 6))
    plt.subplot(121)
    plt.title('Precision-Recall Curve')
    for i in range(len(pred_funcs)):
        plt.plot(recall[i], precision[i], color=colors[i], label='%s' % (algorithms[i]))
    plt.plot([0, 1], [0, 1], color='red', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower center")

    plt.subplot(122)
    plt.title('Receiver Operating Characteristic')
    for i in range(len(pred_funcs)):
        plt.plot(fpr[i], tpr[i], color=colors[i], label='%s (area = %0.2f)' % (algorithms[i], roc_auc[i]))
    plt.text(x=.4, y=.5, s='AUC', fontsize=20, bbox=dict(facecolor='red', alpha=0.5))
    plt.plot([0, 1], [0, 1], color='red', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('PR&ROC Analysis.png')


def main():
    # data_training, target_training = ff.load_original_dataset('data.txt')
    data_training, target_training = ff.load_dataset('training.txt')
    data_test, target_test = ff.load_dataset('test.txt')
    data_pca = get_principal_component(data_training)
    pred_funcs = []
    lr = LogisticRegression()
    lr.fit(data_pca, target_training)
    pred_funcs.append(lr.predict)
    svc = SVC()
    svc.fit(data_pca, target_training)
    pred_funcs.append(svc.predict)
    rfc = RandomForestClassifier()
    rfc.fit(data_pca, target_training)
    pred_funcs.append(rfc.predict)
    dtc = DecisionTreeClassifier()
    dtc.fit(data_pca, target_training)
    pred_funcs.append(dtc.predict)
    plot_decision_boundary(pred_funcs, data_pca, target_training)
    plot_pr_roc(pred_funcs, get_principal_component(data_test), target_test)


def test():
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    dataset_training = np.array(
        pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'))
    data_training, target_training = dataset_training[:, :-1], dataset_training[:, -1]

    estimator = PCA(n_components=2)
    data_training_pca = estimator.fit_transform(data_training)

    def plot_scatter(data, target):
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'gray']
        for i in range(len(colors)):
            x1 = data[:, 0][target == i]
            x2 = data[:, 1][target == i]
            plt.scatter(x1, x2, c=colors[i])
        plt.title('Handwritten Digits')
        plt.legend(range(10))
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.show()

    plot_scatter(data_training_pca, target_training)


if __name__ == '__main__':
    main()
    # test()
