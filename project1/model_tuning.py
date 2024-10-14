from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
import joblib

class ModelTuner:
    def __init__(self, model, param_grid, search_method='grid', cv=5, random_state=42, n_iter=50):
        """
        Initialize the ModelTuner with the specified parameters.
        
        :param model: The machine learning model to tune.
        :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try.
        :param search_method: The search method to use ('grid', 'random', 'bayes').
        :param cv: Number of cross-validation folds.
        :param random_state: Random state for reproducibility.
        :param n_iter: Number of parameter settings that are sampled in RandomizedSearchCV or BayesSearchCV.
        """
        self.model = model
        self.param_grid = param_grid
        self.search_method = search_method
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter
        self.search = None

    def tune(self, X_train, y_train):
        """Perform hyperparameter tuning based on the specified search method."""
        if self.search_method == 'grid':
            self.search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)
        elif self.search_method == 'random':
            self.search = RandomizedSearchCV(
                self.model, self.param_grid, cv=self.cv, scoring='accuracy',
                n_jobs=-1, random_state=self.random_state, n_iter=self.n_iter
            )
        elif self.search_method == 'bayes':
            self.search = BayesSearchCV(
                self.model, self.param_grid, cv=self.cv, scoring='accuracy',
                n_jobs=-1, n_iter=self.n_iter, random_state=self.random_state
            )
        else:
            raise ValueError("search_method must be 'grid', 'random', or 'bayes'")
        
        print(f"Starting {self.search_method} search for hyperparameters...")
        self.search.fit(X_train, y_train)
        print(f"Best parameters found: {self.search.best_params_}")
        return self.search.best_estimator_

    def save_history(self, filename):
        """Save the hyperparameter tuning history."""
        joblib.dump(self.search.cv_results_, filename)
        print(f"Tuning history saved to {filename}")