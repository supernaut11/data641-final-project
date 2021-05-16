import datetime

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
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

def knn_experiment(X_train, y_train, X_test, y_test, plot_metrics=False):
    print(f'running knn classifier grid search')

    label_normalizer = lambda x: 1 if x == 'y' else 0
    y_train = [y for y in map(label_normalizer, y_train)]
    y_test = [y for y in map(label_normalizer, y_test)]

    knn_classifier = KNeighborsClassifier()
    
    scoring = 'average_precision'
    gsearch = GridSearchCV(knn_classifier, 
        {
            'n_neighbors': range(1,31), 
            'weights': ['uniform', 'distance'],
            'p': [1,2]
        },
        scoring=scoring)

    gsearch.fit(X_train, y_train)

    predicted_labels = gsearch.best_estimator_.predict(X_test)

    print(f'best knn hyperparameters: {gsearch.best_params_}')
    print(f'best {scoring} score: {gsearch.best_score_}')

    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in [1, 0]:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall for label {} = {}'.format(label, metrics.recall_score(predicted_labels, y_test, pos_label=label)))
    
    if plot_metrics:
        print("Generating plots")
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metrics.plot_confusion_matrix(gsearch.best_estimator_, X_test, y_test, 
            cmap='Blues', display_labels=['n', 'y'])
        plt.savefig(f'{now}-knn_exp_conf_matrix.png')
        metrics.plot_roc_curve(gsearch.best_estimator_, X_test, y_test)
        plt.savefig(f'{now}-knn_exp_roc.png')
        plt.show()