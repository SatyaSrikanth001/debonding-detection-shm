from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from src.config import CONFIG

def build_stacked_model():
    """Build stacked ensemble model as per the research paper"""
    model_cfg = CONFIG['models']
    
    # Base models
    estimators = [
        ('svm', SVC(**model_cfg['svm'])),
        ('rf', RandomForestClassifier(**model_cfg['rf'])),
        ('ada', AdaBoostClassifier(**model_cfg['ada'])),
        ('gb', GradientBoostingClassifier(**model_cfg['gb'])),
        ('knn', KNeighborsClassifier(**model_cfg['knn']))
    ]
    
    # Meta model
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method='auto',
        cv=5
    )
    
    return stack