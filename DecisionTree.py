from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree

class DecisionTree:
    def __init__(self):
        self.max_depth = None
        self.criterion = None
        self.leaves = []
        self.dt = None

    def build_tree(self, X_train, y_train, X_val, y_val):
        criterion, max_depth = self.grid_search(X_val, y_val)
        self.dt = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
        self.dt.fit(X_train, y_train)
        return self.dt
    
    def predict(self, X, y):
        return self.dt.predict(X, y)
    
    def predict_proba(self, X):
        return self.dt.predict_proba(X)
    
    def grid_search(self, X, y):
        criterion = ['gini', 'entropy']
        max_depth = [2,4,6,8,10,12]
        parameters = dict(d_tree__criterion=criterion, d_tree__max_depth=max_depth)
        d_tree = tree.DecisionTreeClassifier()
        pipeline = Pipeline(steps=[('d_tree', d_tree)])
        grid = GridSearchCV(pipeline, parameters)
        grid.fit(X, y)
        return grid.best_estimator_.get_params()['d_tree__criterion'], grid.best_estimator_.get_params()['d_tree__max_depth']