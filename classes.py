import pandas as pd
from typing import List, Literal, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, LabelEncoder, normalize
from sklearn.impute import SimpleImputer

#############################################################################################

class NewColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to create new columns in a pandas DataFrame by performing specified operations on existing columns.

    Attributes:
        columns (List[str]): List of two column names to perform the operation on.
        operation (Literal['add', 'divide']): The operation to perform. Can be 'add' or 'divide'.
        new_column_name (str): The name of the newly created column.

    Methods:
        fit(X, y=None):
            Fit method required by scikit-learn BaseEstimator. This method doesn't do anything as this transformer doesn't require fitting.
            Returns self.

        transform(X: pd.DataFrame) -> pd.DataFrame:
            Transform method to create a new column in the input DataFrame X by performing the specified operation on the specified columns.
            Returns a new DataFrame with the new column added.
    """

    def __init__(self, columns: List[str], operation: Literal['add', 'divide'], new_column_name: str):
        self.columns = columns
        self.operation = operation
        self.new_column_name = new_column_name

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        if self.operation == 'add':
            X_new[self.new_column_name] = X[self.columns[0]] + X[self.columns[1]]
        elif self.operation == 'divide':
            X_new[self.new_column_name] = X[self.columns[0]] / X[self.columns[1]]
            X_new[self.new_column_name].fillna(0, inplace=True)
        else:
            raise ValueError("Unsupported operation. Use 'add' or 'divide'.")
        
        return X_new

#############################################################################################

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
A custom transformer to drop specified columns from a pandas DataFrame.

Attributes:
    columns_to_drop (list or str): List of column names or a single column name to drop from the DataFrame.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. Returns self.

    transform(X):
        Transform method to drop specified columns from the input DataFrame X.
        Raises a TypeError if X is not a pandas DataFrame.
        Returns a new DataFrame without the specified columns.
    """

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input should be a pandas DataFrame")
        return X_new.drop(columns=self.columns_to_drop, errors='ignore')
    
#############################################################################################

class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
A custom transformer to remove duplicate rows from a pandas DataFrame.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. Returns self.

    transform(X):
        Transform method to remove duplicate rows from the input DataFrame X.
        Raises a TypeError if X is not a pandas DataFrame.
        Returns a new DataFrame with duplicate rows removed.
    """

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input should be a pandas DataFrame")
        return X_new.drop_duplicates()

#############################################################################################

class ConditionalRowRemover(BaseEstimator, TransformerMixin):
    """
A custom transformer to remove rows from a pandas DataFrame based on specified conditions.

Attributes:
    conditions (dict): Dictionary where keys are column names and values are the desired values for those columns.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. Returns self.

    transform(X):
        Transform method to remove rows from the input DataFrame X that meet specified conditions.
        Raises a TypeError if X is not a pandas DataFrame.
        Returns a new DataFrame with rows removed based on the conditions.
    """

    def __init__(self, conditions):
        self.conditions = conditions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input should be a pandas DataFrame")
        mask = pd.Series([False] * len(X_new), index=X_new.index)
        for column, value in self.conditions.items():
            if column in X_new.columns:
                mask |= (X_new[column] == value)
        filtered_data = X_new[~mask].copy()
        return filtered_data

#############################################################################################

class CustomImputer(BaseEstimator, TransformerMixin):
    """
A custom transformer to impute missing values in specified columns of a pandas DataFrame.

Attributes:
    strategy (str, optional): Strategy to use for imputation. Possible values: "mean", "median", "most_frequent", or "constant". Default is "mean".
    columns (list of str, optional): List of column names to impute. If None, imputes all columns. Default is [].
    imputer (SimpleImputer): Instance of SimpleImputer from scikit-learn used for imputation.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. Fits the imputer on the specified columns in the input DataFrame X.
        Returns self.

    transform(X):
        Transform method to impute missing values in the specified columns of the input DataFrame X.
        Returns a new DataFrame with missing values imputed using the specified imputation strategy.
    """

    def __init__(self, strategy="mean", columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.columns = columns if columns is not None else []
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])
        return X_transformed
    
#############################################################################################

class OutlierRemover_IQR(BaseEstimator, TransformerMixin):
    """
    A custom transformer to remove outliers from numeric columns of a pandas DataFrame.

    Attributes:
        threshold (float, optional): Multiplier for the IQR to define outliers. Default is 1.5.

    Methods:
        fit(X, y=None):
            Fit method required by scikit-learn BaseEstimator. Returns self.

        transform(X, y=None):
            Transform method to remove outliers from the numeric columns of the input DataFrame X.
            Computes IQR for numeric columns, identifies outliers based on the threshold,
            and returns a new DataFrame with outlier rows removed.
    """

    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.copy()

        numeric_columns = X_new.select_dtypes(include=[np.number]).columns
        Q1 = X_new[numeric_columns].quantile(0.25)
        Q3 = X_new[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (self.threshold * IQR)
        upper_bound = Q3 + (self.threshold * IQR)
        
        outlier_mask = ((X_new[numeric_columns] < lower_bound) | (X_new[numeric_columns] > upper_bound)).any(axis=1)
        cleaned_data = X_new[~outlier_mask]
        return cleaned_data
    
#############################################################################################

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
A custom transformer to apply label encoding on specified columns of a pandas DataFrame.

Attributes:
    columns (list of str): List of column names to apply label encoding.
    encoders (dict): Dictionary to store LabelEncoder instances for each column.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. Fits LabelEncoder on the specified columns in the input DataFrame X.
        Returns self.

    transform(X):
        Transform method to apply label encoding on the specified columns of the input DataFrame X.
        Returns a new DataFrame with label encoded values for the specified columns.
    """

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[column])
        return X_transformed
    
#############################################################################################

class CustomPowerTransformer(BaseEstimator, TransformerMixin):
    """
A custom transformer to apply power transformation on specified columns of a pandas DataFrame.

Attributes:
    columns (list of str): List of column names to apply power transformation.
    standardize (bool): Whether to standardize the transformed data. Default is True.
    transformers (dict): Dictionary to store PowerTransformer instances for each column.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. Fits PowerTransformer on the specified columns in the input DataFrame X.
        Returns self.

    transform(X):
        Transform method to apply power transformation on the specified columns of the input DataFrame X.
        Returns a new DataFrame with transformed values for the specified columns.
    """

    def __init__(self, columns: List[str], standardize: bool = True):
        self.columns = columns
        self.standardize = standardize
        self.transformers = {}

    def fit(self, X, y=None):
        for column in self.columns:
            transformer = PowerTransformer(standardize=self.standardize)
            transformer.fit(X[[column]])
            self.transformers[column] = transformer
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            transformer = self.transformers[column]
            X_transformed[column] = transformer.transform(X[[column]])
        return X_transformed
    
#############################################################################################

class NormalizeTransformer(BaseEstimator, TransformerMixin):
    """
A custom transformer to normalize a pandas DataFrame using sklearn's normalize function.

Attributes:
    norm (str): Norm to use for normalization. Can be 'l1', 'l2', or 'max'. Default is 'l2'.

Methods:
    fit(X, y=None):
        Fit method required by scikit-learn BaseEstimator. This method doesn't do anything as normalization doesn't require fitting.
        Returns self.

    transform(X):
        Transform method to apply normalization on the input DataFrame X using the specified norm.
        Returns a new DataFrame with normalized values.
    """

    def __init__(self, norm='l2'):
        self.norm = norm

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        X_normalized = normalize(X_new, norm=self.norm)
        return pd.DataFrame(X_normalized, columns=X_new.columns)