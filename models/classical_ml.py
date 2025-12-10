import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def load_data():
    '''
    Load train and test sets from .npz files
    '''
    train = np.load('./data/processed/train.npz')
    test = np.load('./data/processed/test.npz')

    X_train = train['X']
    y_train = train['y']
    class_names = train['class_names']

    X_test = test['X']
    y_test = test['y']

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

def run_model(X_train, y_train, X_test, y_test):
    '''
    Run Logistic Regression classifier 
    '''
    # Flatten
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # PCA
    pca = PCA(n_components=50, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = LogisticRegression(multi_class='multinomial', max_iter=200, n_jobs=-1, random_state=42)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca) # Make predictions

    evaluate(y_test, y_pred)

def main():
    X_train, y_train, X_test, y_test, class_names = load_data()
    run_model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
