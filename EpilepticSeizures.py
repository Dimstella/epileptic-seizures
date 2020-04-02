import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier


def dataset_info(dataset):
    print('Table', dataset.head())
    print('Column names', dataset.columns)
    print(dataset.info())
    print('Number of columns and rows', dataset.shape)

    # Dataset correlation
    dc = dataset[1::].corr()
    sns.heatmap(dc, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()


def data_processing(dataset):
    dataset["y"] = dataset.y == 1  # change the y column to binary and set 1 for seizure and 0 for no seizure results
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # Data visualization
    cols = dataset.columns
    tgt = dataset.y
    ax = sns.countplot(tgt, label="Count")
    plt.show()

    plt.subplot(511)
    plt.plot(X[1, :])
    plt.title('Classes')
    plt.ylabel('uV')
    plt.subplot(512)
    plt.plot(X[7, :])
    plt.subplot(513)
    plt.plot(X[12, :])
    plt.subplot(514)
    plt.plot(X[0, :])
    plt.subplot(515)
    plt.plot(X[2, :])
    plt.xlabel('Samples')
    plt.show()

    # Taking care of missing data
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[:, 0::])
    X[:, 0::] = imputer.transform(X[:, 0::])

    # Encoding categorical data
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


def classifier(classifier, X_train, X_test, y_train, y_test):
    # Fitting classifier to the Training set
    classifier = classifier
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix')
    print(cm)

    return y_pred

def CalculateAccuracy(y_pred, classifier, X_test, y_test, title):

    # Apply Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Calculate accuracy with threshhold 0.5
    thresh = 0.5
    accuracy = accuracy_score(y_test, (y_pred > thresh))

    # Applying k-Fold cross validation
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test, cv=10)
    ma = accuracies.mean()
    plt.plot(accuracies)
    plt.title(title)
    plt.show()

    return mse, accuracy, ma

def PCAfunc(X_train, X_test):
    pca = PCA(.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    print('PCA: ', explained_variance)

    return X_train, X_test

def printResults(L1, L2, firstCL, secondCL, firstPCA, secondPCA):

    data = [{L1: firstCL[0], L2: secondCL[0]}, {L1:firstCL[1], L2: secondCL[1]}, {L1:firstCL[2], L2: secondCL[2]},
            {L1: firstPCA[0], L2: secondPCA[0]}, {L1:firstPCA[1], L2: secondPCA[1]}, {L1:firstPCA[2], L2: secondPCA[2]}]

    CrossVal = pd.DataFrame(data, index =['Mean Square Error', 'Accuracy Score', 'k-Fold',
                                          'Mean Square Error after PCA', 'Accuracy Score after PCA', 'k-Fold after PCA'])

    print(CrossVal)



Epileptic_Seizures = pd.read_csv("data.csv", sep=",")


print("------------------------------- Data info and preprocessing -------------------------------")
dataset_info(Epileptic_Seizures)
DP = data_processing(Epileptic_Seizures)


print("------------------------------- Naive Bayes -------------------------------")
pred_NB = classifier(GaussianNB(), DP[0], DP[1], DP[2], DP[3])
n = CalculateAccuracy(pred_NB, GaussianNB(), DP[1], DP[3], 'k-Fold cross validation for Naive Bayes')

print("------------------------------- Linear Classifier -------------------------------")
pred_LC = classifier(SGDClassifier(alpha=0.1, random_state=42),  DP[0], DP[1], DP[2], DP[3])
ln = CalculateAccuracy(pred_LC, SGDClassifier(alpha=0.1, random_state=42), DP[1], DP[3], 'k-Fold cross validation for Linear Classifier')

print("------------------------------- Results after PCA -------------------------------")
pcaVar = PCAfunc(DP[0], DP[1])
print('Naive Bayes')
pred_NB = classifier(GaussianNB(), pcaVar[0], pcaVar[1], DP[2], DP[3])
nPCA = CalculateAccuracy(pred_NB, GaussianNB(), pcaVar[1], DP[3], 'k-Fold cross validation for Naive Bayes after PCA')
print('Linear Classifier')
pred_LC = classifier(SGDClassifier(alpha=0.1, random_state=42),  pcaVar[0], pcaVar[1], DP[2], DP[3])
lnPCA = CalculateAccuracy(pred_LC, SGDClassifier(alpha=0.1, random_state=42), pcaVar[1], DP[3], 'k-Fold cross validation for Linear Classifier after PCA')

printResults('Naive Bayes', 'Linear Classifier', n, ln, nPCA, lnPCA)

