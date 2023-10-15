import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below

  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict

  #define fit to do nothing but give warning
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  #write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
    #your assert code below

    column_set = set(X.columns)
    not_found = set(self.mapping_dict.keys()) - column_set
    assert not not_found, f"Columns {not_found}, are not in the data table"

    X_ = X.copy()
    return X_.rename(columns=self.mapping_dict)

  #write fit_transform that skips fit
  def fit_transform(self, X, y = None):
    #self.fit(X,y)  #no need for fit method
    result = self.transform(X)
    return result


class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result


class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  #fill in the rest below
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column {self.target_column}'
    X_ = X.copy()
    X_ = pd.get_dummies(X_, columns=[self.target_column],
                        dummy_na=self.dummy_na,
                        drop_first = self.drop_first)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.targ_col = target_column
    self.fitted = False
    self.left = 0
    self.right = 0
    self.min = 0
    self.max = 0

  #define methods below
  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.targ_col in X.columns, f'Column Error: Target Column "{self.targ_col}" not present in given dataframe.'

    self.mean = X[self.targ_col].mean()
    self.sigma = X[self.targ_col].std()

    self.left = self.mean - (3 * self.sigma)
    self.right = self.mean + (3 * self.sigma)

    self.min = X[self.targ_col].max()
    self.max = X[self.targ_col].min()

    self.fitted = True

    return self

  def transform(self, X):
    assert self.fitted , f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    copy = X.copy()
    copy[self.targ_col] = copy[self.targ_col].clip(lower=self.left, upper=self.right)
    copy.reset_index(inplace=True, drop=True)

    return copy

  def fit_transform(self, X, y = None):
    self.fit(X, y)
    return self.transform(X)

class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    self.threshold = threshold
    self.correlated_columns = None
    self.transformed = False

  #define methods below
  def fit(self, X, y = None):
    self.X = X.corr(method='pearson')
    self.X = self.X.abs() > self.threshold

    self.X = self.X.where(np.triu(np.full(self.X.shape, True), k = 1), False)

    self.correlated_columns = [col for col in self.X.columns if any(self.X[col])]

    self.transformed = True

    return self

  def transform(self, X):
    assert self.transformed, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    copy = X.copy()
    return copy.drop(columns=self.correlated_columns)


  def fit_transform(self, X, y = None):
    self.fit(X, y)
    return self.transform(X)



class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']

    self.fence = fence
    self.targ_col = target_column
    self.fitted = False

    self.fence_left = 0
    self.fence_right = 0
  
  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.targ_col in X.columns, f'Column Error: Target Column "{self.targ_col}" not present in given dataframe.'

    inner_left = q1 = X[self.targ_col].quantile(0.25)
    inner_right = q3 = X[self.targ_col].quantile(0.75)
    iqr = q3-q1

    inner_left = q1-1.5*iqr
    inner_right = q3+1.5*iqr
    outer_left = q1-3*iqr
    outer_right = q3+3*iqr

    if self.fence == 'inner':
      self.fence_left = inner_left
      self.fence_right = inner_right
    
    elif self.fence == 'outer':
      self.fence_left = outer_left
      self.fence_right = outer_right
    
    self.fitted = True
    return self


  def transform(self, X):
    assert self.fitted , f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    copy = X.copy()
    copy[self.targ_col] = copy[self.targ_col].clip(lower=self.fence_left, upper=self.fence_right)
    copy.reset_index(inplace=True, drop=True)

    return copy

  def fit_transform(self, X, y = None):
    self.fit(X, y)
    return self.transform(X)


class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    #fill in rest below
    self.col = column
    self.fitted = False
    self.med = 0
    self.iqr = 0
  
  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.col in X.columns, f'Column Error: Target Column "{self.col}" not present in given dataframe.'

    self.iqr = X[self.col].quantile(.75) - X[self.col].quantile(.25)
    self.med = X[self.col].median()

    self.fitted = True
    return self

  def transform(self, X):
    assert self.fitted , f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

    copy = X.copy()
    copy[self.col] -= med
    copy[self.col] /= iqr

    return copy

  def fit_transform(self, X, y = None):
    self.fit(X, y)
    return self.transform(X)
