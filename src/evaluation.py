# Data Handling
import numpy as np
from collections import Counter

# Model Utilities
from sklearn.base import clone
from sklearn.pipeline import Pipeline

# Preprocessing & Validation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

#Sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def sampling_evaluate(X, y, classifiers, names, cost_matrix, sampler=None, splits=5):
    """
    Evaluates classifiers using a specific sampling strategy and a cost matrix.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        classifiers: A list of scikit-learn classifier objects
        names: The string names for the classifiers for reporting
        cost_matrix: 2x2 array defining costs of FP and FN.
        sampler: An imblearn sampler object
        splits: Number of folds for StratifiedKFold.
    Returns:
        final_summary: Dictionary of model performance.
        fold_matrices: Aggregated confusion matrices.
    """
    
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    fold_loss = {name: [] for name in names} 
    fold_matrices = {name: np.zeros((2, 2)) for name in names}
    fold_accuracies = {name: [] for name in names}
    final_summary = {}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if sampler is not None:
            counts = Counter(y_train)
            majority_class = max(counts, key=counts.get) 
            minority_class = min(counts, key=counts.get) 
    
            n_maj = counts[majority_class]
            n_min = counts[minority_class]
            
            target_maj = int(n_maj * sampler['majority']) 
            target_min = int(n_min * sampler['minority'])

            rus = RandomUnderSampler(sampling_strategy={'good': target_maj, 'bad': n_min}, random_state=42)
            X_mid, y_mid = rus.fit_resample(X_train_scaled, y_train)
            
            ros = RandomOverSampler(sampling_strategy={'good': target_maj, 'bad': target_min}, random_state=42)
            X_res, y_res = ros.fit_resample(X_mid, y_mid)
        else:
            X_res, y_res = X_train_scaled, y_train

        for name, clf in zip(names, classifiers):
            model = clone(clf)
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test_scaled)
            conf_m = confusion_matrix(y_test, y_pred, labels = ['good', 'bad'])
            fold_matrices[name] += conf_m
 
            loss = np.sum(conf_m * cost_matrix) 
            fold_loss[name].append(loss)

            acc = accuracy_score(y_test, y_pred)
            fold_accuracies[name].append(acc)
    
    for name in names:
        avg_loss = np.mean(fold_loss[name])
        std_loss = np.std(fold_loss[name])
        avg_acc = np.mean(fold_accuracies[name])
        final_summary[name] = {"loss": avg_loss, "accuracy": round(avg_acc, 2), "std": round(std_loss,2)}
        print(f"{name} → Average loss: {avg_loss:.2f} | Accuracy: {avg_acc:.2%}")
    return final_summary, fold_matrices

def weights_evaluate(X, y, classifiers, names, cost_matrix, weight_bad=5, weight_good=1, splits = 5):
    """
    Evaluates classifiers using sample weights during training.

    Args:
        X: Feature DataFrame
        y: Target Series
        classifiers: A list of scikit-learn classifier objects
        names: The string names for the classifiers for reporting
        cost_matrix: 2x2 array defining costs of FP and FN.
        weight_bad: Penalty multiplier applied to samples labeled 'bad'
        weight_good: Penalty multiplier applied to samples labeled 'good'
        splits: Number of folds for StratifiedKFold.
    Returns:
        final_summary: Dictionary of model performance.
        fold_matrices: Aggregated confusion matrices.

    """
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    fold_loss = {name: [] for name in names} 
    fold_matrices = {name: np.zeros((2, 2)) for name in names}
    fold_accuracies = {name: [] for name in names}
    final_summary = {}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        weights = np.ones(len(y_train))

        weights[y_train == 'bad'] = weight_bad
        weights[y_train == 'good'] = weight_good

        for name, clf in zip(names, classifiers):
            model = clone(clf)
            model.fit(X_train_scaled, y_train, sample_weight=weights)
            y_pred = model.predict(X_test_scaled)
            conf_m = confusion_matrix(y_test, y_pred, labels = ['good','bad'])
            fold_matrices[name] += conf_m
            loss = np.sum(conf_m * cost_matrix)
            fold_loss[name].append(loss)

            acc = accuracy_score(y_test, y_pred)
            fold_accuracies[name].append(acc)

    for name in names:
        avg_loss = np.mean(fold_loss[name])
        std_loss = np.std(fold_loss[name])
        avg_acc = np.mean(fold_accuracies[name])
        final_summary[name] = {"loss": avg_loss, "accuracy": round(avg_acc, 2), "std": round(std_loss,2)}
        print(f"{name} → Average loss: {avg_loss:.2f} | Accuracy: {avg_acc:.2%}")
    return final_summary, fold_matrices

def calibration_evaluate(X, y, mapping, clfs, names, cost_matrix, calibrate=None, n_splits=5):
    """
    Evaluates classifiers using probability calibration and Bayes Risk Minimization.
    
    This method optimizes the decision threshold by calculating the expected cost 
    for each class. Instead of predicting the class with the highest probability, 
    it predicts the class that minimizes total financial risk based on the cost_matrix.

    Args:
        X: Feature DataFrame
        y: Target Series
        mapping: Dictionary to map original labels to integers
        clfs: A list of scikit-learn classifier objects
        names: The string names for the classifiers for reporting
        cost_matrix: 2x2 array defining costs of FP and FN.
        calibrate: Calibration method
        n_splits: Number of folds for StratifiedKFold.
    Returns:
        final_summary: Dictionary of model performance.
        fold_matrices: Aggregated confusion matrices.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_loss = {name: [] for name in names} 
    fold_matrices = {name: np.zeros((2, 2)) for name in names}
    fold_accuracies = {name: [] for name in names}
    final_summary = {}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_train = np.array([mapping[val] for val in y_train])
        y_test  = np.array([mapping[val] for val in y_test])
    
        for name, clf in zip(names, clfs):
            base_clf = clone(clf)
            if calibrate is None:
                model = base_clf.fit(X_train_scaled, y_train)
            else:
                model = CalibratedClassifierCV(
                    base_clf,
                    method=calibrate,
                    cv=3
                ).fit(X_train_scaled, y_train)
            y_pred_prob = model.predict_proba(X_test_scaled)
            y_pred = np.argmin(
                np.matmul(y_pred_prob, np.array(cost_matrix)),
                axis=1)
            conf_m = confusion_matrix(
                y_test, y_pred, labels = [0, 1])
            fold_matrices[name] += conf_m
            loss = np.sum(conf_m * cost_matrix)
            fold_loss[name].append(loss)

            acc = accuracy_score(y_test, y_pred)
            fold_accuracies[name].append(acc)

    print(f"\n=== Calibration: {calibrate if calibrate else 'none'} ===")
    for name in names:
        avg_loss = np.mean(fold_loss[name])
        std_loss = np.std(fold_loss[name])
        avg_acc = np.mean(fold_accuracies[name])
        final_summary[name] = {"loss": avg_loss, "accuracy": round(avg_acc, 2), "std": round(std_loss,2)}
        print(f"{name} → Average loss: {avg_loss:.2f} | Accuracy: {avg_acc:.2%}")
    return final_summary, fold_matrices