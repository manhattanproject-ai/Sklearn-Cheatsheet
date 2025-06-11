# Sklearn-Cheatsheet for Developers

## Introduction-What-is-Sklearn?

> NumPy (Numerical Python) is the fundamental library for numerical computing in Python, providing powerful capabilities for working with large, multi-dimensional arrays and matrices. At its core, NumPy introduces the ndarray object, which is significantly more efficient for storing and manipulating numerical data than standard Python lists, especially for large datasets. It offers a vast collection of high-level mathematical functions to operate on these arrays, covering linear algebra, Fourier transforms, random number generation, and more. NumPy's efficiency stems from its implementation in C and Fortran, allowing for high-performance operations that are crucial for scientific computing, data analysis, and machine learning, making it an indispensable building block for many other Python libraries like Pandas and Scikit-learn.


## 1. Importing Scikit-learn Modules

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|`import sklearn`|	Imports the entire Scikit-learn library. Rarely used directly, as specific modules are typically imported.|
|`from sklearn.model_selection import train_test_split`|	Imports the train_test_split function from the model_selection module, used for splitting data into training and testing sets.|
|`from sklearn.preprocessing import StandardScaler`|	Imports StandardScaler from the preprocessing module, used for standardizing features by removing the mean and scaling to unit variance.|
|`from sklearn.linear_model import LogisticRegression`|	Imports LogisticRegression from the linear_model module, a common classification algorithm.|
|`from sklearn.svm import SVC`|	Imports SVC (Support Vector Classifier) from the svm module, another classification algorithm.|
|`from sklearn.tree import DecisionTreeClassifier`|	Imports DecisionTreeClassifier from the tree module, for classification based on decision trees.|
|`from sklearn.ensemble import RandomForestClassifier`|	Imports RandomForestClassifier from the ensemble module, an ensemble classification method.|
|`from sklearn.metrics import accuracy_score`|	Imports accuracy_score from the metrics module, used for evaluating model performance.|
|`from sklearn.cluster import KMeans`|	Imports KMeans from the cluster module, a popular clustering algorithm.|
|`from sklearn.decomposition import PCA`|	Imports PCA (Principal Component Analysis) from the decomposition module, for dimensionality reduction.|
|`from sklearn.pipeline import Pipeline`|	Imports Pipeline from the pipeline module, used to chain multiple processing steps together.|
|`from sklearn.datasets import load_iris`|	Imports load_iris from the datasets module, used for loading sample datasets built into Sklearn.|
|`from sklearn.impute import SimpleImputer`|	Imports SimpleImputer from the impute module, for handling missing values.|
|`from sklearn.model_selection import GridSearchCV`|	Imports GridSearchCV from the model_selection module, for systematic hyperparameter tuning.|

## 2. Data Loading & Splitting

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|`from sklearn.datasets import load_dataset_name`|	Importing Built-in Datasets: Imports specific datasets bundled with scikit-learn for educational and example purposes (e.g., load_iris, load_breast_cancer, load_digits). These functions typically return a Bunch object, which is like a dictionary with attributes.|
|`data_bunch = load_iris()`|	Loads a built-in dataset into a Bunch object. The features are usually in data_bunch.data (NumPy array) and targets in data_bunch.target (NumPy array).|
|`X = data_bunch.data`|	Accesses the feature matrix (independent variables) from a loaded Bunch object.|
|`y = data_bunch.target`|	Accesses the target vector (dependent variable) from a loaded Bunch object.|
|`X = pd.DataFrame(data_bunch.data, columns=data_bunch.feature_names)`|	Converts the feature data from a NumPy array to a Pandas DataFrame, using provided feature names. This is often good practice for better readability and manipulation.|
|`from sklearn.model_selection import train_test_split`|	Importing Data Splitting Utility: Imports the primary function for splitting datasets into training and testing subsets.|
|`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`|	Basic Train-Test Split: Splits the feature matrix X and target vector y into training and testing sets. test_size specifies the proportion of the dataset to include in the test split (e.g., 0.2 for 20%). random_state ensures reproducibility of the split.|
|`test_size	(Parameter)`| A float between 0.0 and 1.0 (proportion of test set) or an int (absolute number of test samples).|
|`train_size	(Parameter)`| A float between 0.0 and 1.0 (proportion of train set) or an int (absolute number of train samples). Default is complement of test_size.|
|`random_state	(Parameter)`| An integer or RandomState instance. Used for shuffling the data before splitting. Passing an int fixes the shuffle order, ensuring the same split each time.|
|`shuffle=True	(Parameter)`| Boolean, default True. Whether or not to shuffle the data before splitting. Generally keep True.|
|`stratify=y	(Parameter)`| An array-like. If not None, data is split in a stratified fashion, using y as the class labels. This is crucial for classification tasks to maintain the same proportion of classes in train and test sets.|
|`X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)`|	Splitting Features Only: If you only have features and no separate target variable, you can split X alone.|
|`from sklearn.model_selection import KFold`|	Importing K-Fold Cross-Validation: Imports the class for K-Fold cross-validation, a technique for robust model evaluation.|
|`kf = KFold(n_splits=5, shuffle=True, random_state=42)`|	Creates a KFold object to generate indices to split data into train/test sets. n_splits is the number of folds (e.g., 5-fold CV).|
|`for train_index, test_index in kf.split(X):`|	Iterates through the folds, yielding train_index and test_index arrays for each fold. These indices can then be used to slice X and y.|
|`X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]`|	(With Pandas) Slices the DataFrame X using the generated indices for one fold.|
|`y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]`|	(With Pandas) Slices the Series y using the generated indices for one fold.|
|`from sklearn.model_selection import StratifiedKFold`|	Importing Stratified K-Fold Cross-Validation: Imports the class for K-Fold cross-validation that preserves the percentage of samples for each class. Essential for imbalanced datasets.|
|`skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`|	Creates a StratifiedKFold object. Similar to KFold but ensures class distribution.|
|`for train_index, test_index in skf.split(X, y):`|	Iterates through the folds, using X and y to ensure stratification during the split.|
