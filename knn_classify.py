import datetime

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier

def knn_classify(X_train, y_train, X_test, y_test, plot_metrics=False, k=5):
    print(f'running knn classifier, k={k}')

    # Create a KNN classifier trained on the featurized training data
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    predicted_labels = knn_classifier.predict(X_test)

    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in ['y', 'n']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall for label {} = {}'.format(label, metrics.recall_score(predicted_labels, y_test, pos_label=label)))

    if plot_metrics:
        print("Generating plots")
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metrics.plot_confusion_matrix(knn_classifier, X_test, y_test, normalize='true')
        plt.savefig(f'{now}-knn_conf_matrix.png')
        metrics.plot_roc_curve(knn_classifier, X_test, y_test)
        plt.savefig(f'{now}-knn_roc.png')
        plt.show()