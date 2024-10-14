import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, data_path, target_column, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        """Load dataset from the specified path."""
        data = pd.read_csv(self.data_path)
        print(f"Data loaded from {self.data_path}")
        return data

    def preprocess(self, data):
        """Preprocess the dataset: handle missing values, encode categorical variables, and scale features."""
        # Drop rows with missing target
        data = data.dropna(subset=[self.target_column])
        
        # Separate features and target
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Define preprocessing pipelines for numerical and categorical data
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        print("Data preprocessed: missing values handled, categorical variables encoded, features scaled.")
        return X_processed, y, preprocessor

    def split_data(self, X, y):
        """Split the dataset into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"Data split into train and test sets with test size = {self.test_size}")
        return X_train, X_test, y_train, y_test