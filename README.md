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
|`test_size`| 	(Parameter) A float between 0.0 and 1.0 (proportion of test set) or an int (absolute number of test samples).|
|`train_size`|	(Parameter) A float between 0.0 and 1.0 (proportion of train set) or an int (absolute number of train samples). Default is complement of test_size.|
|`random_state`|	(Parameter) An integer or RandomState instance. Used for shuffling the data before splitting. Passing an int fixes the shuffle order, ensuring the same split each time.|
|`shuffle=True`|	(Parameter) Boolean, default True. Whether or not to shuffle the data before splitting. Generally keep True.|
|`stratify=y`|	(Parameter) An array-like. If not None, data is split in a stratified fashion, using y as the class labels. This is crucial for classification tasks to maintain the same proportion of classes in train and test sets.|
|`X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)`|	Splitting Features Only: If you only have features and no separate target variable, you can split X alone.|
|`from sklearn.model_selection import KFold`|	Importing K-Fold Cross-Validation: Imports the class for K-Fold cross-validation, a technique for robust model evaluation.|
|`kf = KFold(n_splits=5, shuffle=True, random_state=42)`|	Creates a KFold object to generate indices to split data into train/test sets. n_splits is the number of folds (e.g., 5-fold CV).|
|`for train_index, test_index in kf.split(X):`|	Iterates through the folds, yielding train_index and test_index arrays for each fold. These indices can then be used to slice X and y.|
|`X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]`|	(With Pandas) Slices the DataFrame X using the generated indices for one fold.|
|`y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]`|	(With Pandas) Slices the Series y using the generated indices for one fold.|
|`from sklearn.model_selection import StratifiedKFold`|	Importing Stratified K-Fold Cross-Validation: Imports the class for K-Fold cross-validation that preserves the percentage of samples for each class. Essential for imbalanced datasets.|
|`skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`|	Creates a StratifiedKFold object. Similar to KFold but ensures class distribution.|
|`for train_index, test_index in skf.split(X, y):`|	Iterates through the folds, using X and y to ensure stratification during the split.|

## 3. Data Preprocessing & Transformation

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|`from sklearn.preprocessing import StandardScaler`|	Imports the StandardScaler for scaling features to a standard normal distribution (mean=0, variance=1).|
|`scaler = StandardScaler()`|	Initializes the StandardScaler object.|
|`scaled_data = scaler.fit_transform(X_train)`|	Fits the scaler to the training data X_train and then transforms it. This is the common step for fitting the transformation parameters (mean, std dev) only on the training data.|
|`scaled_test_data = scaler.transform(X_test)`|	Transforms the test data X_test using the parameters learned from the training data. Never fit on test data.|
|`from sklearn.preprocessing import MinMaxScaler`|	Imports the MinMaxScaler for scaling features to a given range (default: 0 to 1).|
|`minmax_scaler = MinMaxScaler()`|	Initializes the MinMaxScaler object.|
|`scaled_data = minmax_scaler.fit_transform(X)`|	Fits the MinMaxScaler to data X and transforms it, scaling features to the 0-1 range.|
|`from sklearn.preprocessing import RobustScaler`|	Imports the RobustScaler for scaling features using statistics that are robust to outliers (median and interquartile range).|
|`robust_scaler = RobustScaler()`|	Initializes the RobustScaler object.|
|`scaled_data = robust_scaler.fit_transform(X)`|	Fits the RobustScaler to data X and transforms it.|
|`from sklearn.preprocessing import Normalizer`|	Imports the Normalizer for scaling individual samples (rows) to unit norm (L1, L2, or max norm). Useful for text features or sparse data.|
|`normalizer = Normalizer(norm='l2')`|	Initializes the Normalizer object, specifying the norm to use (e.g., 'l1', 'l2', 'max').|
|`normalized_data = normalizer.fit_transform(X)`|	Normalizes each row of the input data X.|
|`from sklearn.preprocessing import OneHotEncoder`|	Imports the OneHotEncoder for converting categorical (nominal) features into a one-hot numeric array.|
|`encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)`|	Initializes the OneHotEncoder. handle_unknown='ignore' handles new categories seen during transform. sparse_output=False returns a dense array.|
|`encoded_data = encoder.fit_transform(X_categorical)`|	Fits the encoder to categorical data X_categorical and transforms it.|
|`encoded_test_data = encoder.transform(X_test_categorical)`|	Transforms new categorical data using the fitted encoder.|
|`from sklearn.preprocessing import OrdinalEncoder`|	Imports the OrdinalEncoder for encoding categorical features as ordinal integers. Useful when there's an inherent order.|
|`ordinal_encoder = OrdinalEncoder()`|	Initializes the OrdinalEncoder object.|
|`ordinal_encoded_data = ordinal_encoder.fit_transform(X_categorical)`|	Fits the encoder to categorical data and transforms it into numerical ordinal values.|
|`from sklearn.impute import SimpleImputer`|	Imports SimpleImputer for handling missing values.|
|`imputer = SimpleImputer(strategy='mean')`|	Initializes the SimpleImputer. Common strategies: 'mean', 'median', 'most_frequent', 'constant'.|
|`imputed_data = imputer.fit_transform(X_with_missing)`|	Fits the imputer to data X_with_missing (learning imputation values) and then fills missing values.|
|`from sklearn.impute import KNNImputer`|	Imports KNNImputer for imputing missing values using the k-Nearest Neighbors approach.|
|`knn_imputer = KNNImputer(n_neighbors=5)`|	Initializes the KNNImputer, specifying the number of neighbors to consider.|
|`imputed_data = knn_imputer.fit_transform(X_with_missing)`|	Fills missing values in X_with_missing using the KNN imputation strategy.|
|`from sklearn.preprocessing import PolynomialFeatures`|	Imports PolynomialFeatures for generating polynomial and interaction features.|
|`poly = PolynomialFeatures(degree=2, include_bias=False)`|	Initializes PolynomialFeatures to create polynomial features up to a specified degree. include_bias=False avoids adding an intercept term.|
|`poly_features = poly.fit_transform(X)`|	Generates polynomial features from the input data X.|
|`from sklearn.feature_selection import SelectKBest, f_classif`|	Imports SelectKBest for selecting the top K features based on a scoring function (e.g., f_classif for classification, f_regression for regression).|
|`selector = SelectKBest(f_classif, k=10)`|	Initializes the feature selector to pick the top 10 features using F-score for classification.|
|`selected_features = selector.fit_transform(X, y)`|	Fits the selector to the data X and target y, then transforms X to include only the selected features.|
|`from sklearn.decomposition import PCA`|	Imports PCA (Principal Component Analysis) for dimensionality reduction.|
|`pca = PCA(n_components=2)`|	Initializes PCA to reduce dimensionality to 2 principal components.|
|`X_pca = pca.fit_transform(X)`|	Fits PCA to the data and transforms it into the lower-dimensional space.|

## 4. Supervised Learning: Classification Models

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|`from sklearn.linear_model import LogisticRegression`|	Imports the Logistic Regression classifier, a common linear model for binary and multi-class classification.|
|`model = LogisticRegression(solver='liblinear', random_state=42)`|	Initializes a Logistic Regression model. solver specifies the algorithm for optimization; random_state ensures reproducibility.|
|`from sklearn.tree import DecisionTreeClassifier`|	Imports the Decision Tree classifier, a non-parametric model that learns decision rules from data.|
|`model = DecisionTreeClassifier(max_depth=5, random_state=42)`|	Initializes a Decision Tree classifier. max_depth limits the tree's depth to prevent overfitting.|
|`from sklearn.ensemble import RandomForestClassifier`|	Imports the Random Forest classifier, an ensemble method that builds multiple decision trees and merges their predictions.|
|`model = RandomForestClassifier(n_estimators=100, random_state=42)`|	Initializes a Random Forest classifier. n_estimators specifies the number of trees in the forest.|
|`from sklearn.svm import SVC`|	Imports the Support Vector Classifier, a powerful model for both linear and non-linear classification.|
|`model = SVC(kernel='rbf', C=1.0, random_state=42)`|	Initializes an SVC model. kernel specifies the kernel function (e.g., 'linear', 'poly', 'rbf'); C is the regularization parameter.|
|`from sklearn.neighbors import KNeighborsClassifier`|	Imports the K-Nearest Neighbors classifier, a non-parametric, instance-based learning algorithm.|
|`model = KNeighborsClassifier(n_neighbors=5)`|	Initializes a KNeighborsClassifier. n_neighbors specifies the number of neighbors to consider.|
|`from sklearn.naive_bayes import GaussianNB`|	Imports the Gaussian Naive Bayes classifier, based on Bayes' theorem with a strong independence assumption.|
|`model = GaussianNB()`|	Initializes a Gaussian Naive Bayes model.|
|`from sklearn.ensemble import GradientBoostingClassifier`|	Imports the Gradient Boosting Classifier, an ensemble method that builds trees sequentially, with each tree correcting errors of the previous one.|
|`model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)`|	Initializes a Gradient Boosting Classifier. n_estimators is the number of boosting stages; learning_rate shrinks the contribution of each tree.|
|`from xgboost import XGBClassifier`|	Imports the XGBoost Classifier (requires xgboost installation), a highly efficient and popular gradient boosting framework.|
|`model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)`|	Initializes an XGBoost Classifier. n_estimators is the number of boosting rounds. use_label_encoder=False and eval_metric are often set to suppress warnings in newer versions.|
|`model.fit(X_train, y_train)`|	Trains the classification model. X_train are the training features, y_train are the corresponding training labels.|
|`predictions = model.predict(X_test)`|	Makes class predictions on new data X_test. Returns an array of predicted class labels.|
|`probabilities = model.predict_proba(X_test)`|	Estimates class probabilities for new data X_test. Returns an array where each row sums to 1. (Not all models support this).|
|`score = model.score(X_test, y_test)`|	Calculates the mean accuracy on the given test data and labels.|

## 5. Supervised Learning: Regression Models

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|`from sklearn.linear_model import LinearRegression`|	Import Linear Regression: Imports the class for Ordinary Least Squares Linear Regression.|
|`model = LinearRegression()`|	Instantiate Linear Regression: Creates a Linear Regression model object.|
|`model.fit(X_train, y_train)`|	Train Model: Fits the regression model to the training data. X_train are features, y_train are target values.|
|`y_pred = model.predict(X_test)`|	Make Predictions: Uses the trained model to make predictions on new (test) data X_test.|
|`model.coef_`|	Coefficients: After fitting, returns the estimated coefficients for the features.|
|`model.intercept_`|	Intercept: After fitting, returns the independent term in the linear model.|
|`from sklearn.linear_model import Ridge`|	Import Ridge Regression: Imports the class for Ridge (L2 regularization) Regression, which helps prevent overfitting.|
|`model = Ridge(alpha=1.0)`|	Instantiate Ridge Regression: Creates a Ridge Regression model. alpha is the regularization strength; higher values mean stronger regularization.|
|`from sklearn.linear_model import Lasso`|	Import Lasso Regression: Imports the class for Lasso (L1 regularization) Regression, which can perform feature selection by shrinking coefficients to zero.|
|`model = Lasso(alpha=1.0)`|	Instantiate Lasso Regression: Creates a Lasso Regression model. alpha is the regularization strength.|
|`from sklearn.linear_model import ElasticNet|	Import Elastic Net Regression: Imports the class for Elastic Net Regression, which combines L1 and L2 regularization.|
|`model = ElasticNet(alpha=1.0, l1_ratio=0.5)|	Instantiate Elastic Net: Creates an Elastic Net model. alpha is overall regularization, l1_ratio controls the mix of L1 (Lasso) and L2 (Ridge) penalties (0 for Ridge, 1 for Lasso).|
|`from sklearn.preprocessing import PolynomialFeatures`|	Import Polynomial Features: Imports a transformer for generating polynomial and interaction features.|
|`poly = PolynomialFeatures(degree=2, include_bias=False)`|	Instantiate Poly Features: Creates a transformer to convert features into polynomial terms (e.g., x1,x2,x1^2). degree specifies the maximum degree.|
|`X_poly = poly.fit_transform(X)`|	Transform Data: Applies the polynomial transformation to the data.|
|`from sklearn.tree import DecisionTreeRegressor`|	Import Decision Tree Regressor: Imports the class for Decision Tree-based regression.|
|`model = DecisionTreeRegressor(max_depth=5, random_state=42)`|	Instantiate Decision Tree Regressor: Creates a Decision Tree Regressor. max_depth limits tree growth, random_state ensures reproducibility.|
|`from sklearn.ensemble import RandomForestRegressor`|	Import Random Forest Regressor: Imports the class for Random Forest Regressor, an ensemble method using multiple decision trees.|
|`model = RandomForestRegressor(n_estimators=100, random_state=42)`|	Instantiate Random Forest Regressor: Creates a Random Forest model. n_estimators is the number of trees in the forest.|
|`from sklearn.ensemble import GradientBoostingRegressor`|	Import Gradient Boosting Regressor: Imports the class for Gradient Boosting Regressor, another powerful ensemble method.|
|`model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)`|	Instantiate Gradient Boosting Regressor: Creates a Gradient Boosting model. n_estimators is the number of boosting stages, learning_rate controls the contribution of each tree.|
|`from sklearn.svm import SVR`|	Import Support Vector Regressor: Imports the class for Support Vector Regressor.|
|`model = SVR(kernel='rbf', C=1.0, epsilon=0.1)`|	Instantiate SVR: Creates an SVR model. kernel specifies the kernel type ('linear', 'poly', 'rbf'), C is the regularization parameter, epsilon is the epsilon-tube within which no penalty is associated in the training loss function.|
|`from sklearn.neighbors import KNeighborsRegressor`|	Import K-Nearest Neighbors Regressor: Imports the class for K-Nearest Neighbors Regression.|
|`model = KNeighborsRegressor(n_neighbors=5)`|	Instantiate KNeighborsRegressor: Creates a KNN Regressor. n_neighbors is the number of neighbors to consider.|

## 6. Unsupervised Learning: Clustering Models

> This section covers the standard way to bring the NumPy library into your Python scripts.

|Command | description|
|----------|-------------|
|from sklearn.cluster import KMeans|	Import K-Means: Imports the K-Means clustering algorithm.|
|model = KMeans(n_clusters=k, random_state=seed, ...)|	K-Means Initialization: Creates a K-Means model instance. n_clusters (k) is the number of clusters to form. random_state ensures reproducibility.|
|model.fit(X)|	K-Means Fit: Fits the K-Means model to the data X. The algorithm iteratively assigns data points to clusters and updates cluster centroids.|
|labels = model.predict(X_new)|	K-Means Predict: Predicts the cluster label for new data points X_new based on the fitted centroids.|
|labels = model.fit_predict(X)|	K-Means Fit & Predict (Combined): Fits the model to X and returns the cluster labels for X in one step.|
|centroids = model.cluster_centers_|	K-Means Centroids: After fitting, this attribute holds the coordinates of the cluster centroids.|
|inertia = model.inertia_|	K-Means Inertia (WCSS): The sum of squared distances of samples to their closest cluster center (within-cluster sum of squares). Lower inertia usually means better clustering, but can be misleading for evaluating k.|
|from sklearn.cluster import DBSCAN|	Import DBSCAN: Imports the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.|
|model = DBSCAN(eps=0.5, min_samples=5, ...)|	DBSCAN Initialization: Creates a DBSCAN model instance. eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other. min_samples is the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.|
|labels = model.fit_predict(X)|	DBSCAN Fit & Predict: Fits the DBSCAN model to X and returns the cluster labels for X. Noise points are assigned the label -1.|
|core_samples_mask = np.zeros_like(model.labels_, dtype=bool)&lt;br>core_samples_mask[model.core_sample_indices_] = True	DBSCAN Core Samples: After fitting, model.core_sample_indices_ gives the indices of core samples. This helps identify the dense parts of clusters.
|from sklearn.cluster import AgglomerativeClustering	Import Agglomerative Clustering: Imports hierarchical clustering that performs a bottom-up aggregation.
|model = AgglomerativeClustering(n_clusters=k, linkage='ward', ...)	Agglomerative Clustering Initialization: Creates an instance. n_clusters is the number of clusters to find. linkage specifies the method used to calculate the distance between clusters ('ward', 'complete', 'average', 'single').|
|labels = model.fit_predict(X)	Agglomerative Clustering Fit & Predict: Fits the model to X and returns the cluster labels.|
|from sklearn.cluster import Birch	Import Birch: Imports the BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) algorithm, good for large datasets.|
|model = Birch(n_clusters=k, threshold=0.5, branching_factor=50, ...)	Birch Initialization: Creates a Birch instance. n_clusters is the number of clusters. threshold is the maximum radius of the subcluster. branching_factor is the maximum number of CF subclusters in each node.|
|labels = model.fit_predict(X)	Birch Fit & Predict: Fits the Birch model and returns cluster labels.|
|from sklearn.cluster import MiniBatchKMeans	Import MiniBatchKMeans: Imports a variant of the K-Means algorithm that uses mini-batches to reduce computation time, especially for large datasets.|
|model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=256, ...)	MiniBatchKMeans Initialization: Similar to K-Means, but batch_size determines the size of the mini-batches.|
|labels = model.fit_predict(X)	MiniBatchKMeans Fit & Predict: Fits the model and returns cluster labels.|
|from sklearn.cluster import SpectralClustering	Import Spectral Clustering: Imports a method that uses the eigenvalues of a similarity matrix for dimensionality reduction before clustering.|
|model = SpectralClustering(n_clusters=k, affinity='rbf', ...)	Spectral Clustering Initialization: Creates an instance. affinity defines the similarity matrix (e.g., 'rbf', 'nearest_neighbors').|
|labels = model.fit_predict(X)	Spectral Clustering Fit & Predict: Fits the model and returns cluster labels.|
|from sklearn.metrics import silhouette_score	Import Silhouette Score: Imports a common metric to evaluate the quality of clustering results.|
|score = silhouette_score(X, labels)	Silhouette Score Calculation: Computes the mean Silhouette Coefficient of all samples. A higher score indicates better-defined clusters. Range: -1 to 1.|
|from sklearn.metrics import calinski_harabasz_score	Import Calinski-Harabasz Score: Imports a variance ratio criterion for clustering evaluation.|
|score = calinski_harabasz_score(X, labels)	Calinski-Harabasz Score Calculation: Computes the score. A higher score relates to a model with better defined clusters.|
|from sklearn.metrics import davies_bouldin_score	Import Davies-Bouldin Score: Imports a metric that evaluates the ratio of within-cluster scatter to between-cluster separation.|
|score = davies_bouldin_score(X, labels)	Davies-Bouldin Score Calculation: Computes the score. A lower score relates to a model with better separation between clusters.|



