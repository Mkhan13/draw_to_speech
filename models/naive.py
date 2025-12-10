import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    '''
    Load train and test sets from .npz files
    '''
    train = np.load('./data/processed/train.npz')
    test = np.load('./data/processed/test.npz')

    X_train, y_train = train['X'], train['y']
    X_test, y_test = test['X'], test['y']
    class_names = train['class_names']

    return X_train, y_train, X_test, y_test, class_names

def evaluate(y_true, y_pred):
    '''
    Compute and print evaluation metrics
    '''
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print('Accuracy:', round(accuracy, 4))
    print('Precision:', round(precision, 4))
    print('Recall:', round(recall, 4))
    print('F1-score:', round(f1, 4))

def run_naive_baseline(y_train, y_test):
    '''
    Predict the majority class
    '''
    unique_classes, counts = np.unique(y_train, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]  # Get majority class

    y_pred_test = np.full_like(y_test, majority_class)  # Predict majority class for all

    evaluate(y_test, y_pred_test)

def main():
    X_train, y_train, X_test, y_test, class_names = load_data()
    run_naive_baseline(y_train, y_test)

if __name__ == '__main__':
    main()