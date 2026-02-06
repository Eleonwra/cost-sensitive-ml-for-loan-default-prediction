#Standard libraries
import os

#Data Handling
import pandas as pd
import numpy as np

#Preprocessing
from sklearn.datasets import  fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector

#Machine learning models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt

from src.evaluation import sampling_evaluate, weights_evaluate, calibration_evaluate

def main():
    os.makedirs("plots", exist_ok=True)
    
    X, y = fetch_openml("credit-g", version=1, as_frame=True, parser='auto', return_X_y=True)
    cost_m = [[0, 1], [5, 0]]
    mapping = {'good': 0, 'bad': 1}

    one_hot_encoder = make_column_transformer(
    (OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
     make_column_selector(dtype_include='category')),
    remainder='passthrough')

    X = one_hot_encoder.fit_transform(X)
    X = pd.DataFrame(X)

    sampling_strategies = [ ('baseline', None), ('hybrid_cost_ratio', {'majority': 0.5, 'minority': 5.0})]
    names = ['Random Forest', 'Linear SVM', 'Gaussian NB']
    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(kernel='linear'),
        GaussianNB()]
    
    final_comparison_data = []

    print("\n--- Running Resampling-Based Evaluation ---")
    for sampler_name, sampler in sampling_strategies:
        print(f"\n=== {sampler_name} ===")
        results, fold_matrices = sampling_evaluate(X, y, classifiers, names, cost_m, sampler)
        for name in names:
            final_comparison_data.append({
            "Technique": sampler_name,
            "Model": name,
            "Mean Loss": results[name]["loss"],
            "Std Loss": results[name]["std"],
            "Mean Acc": results[name]["accuracy"]})

    print("\n--- Running Weight-Based Evaluation ---\n")

    results, fold_matrices = weights_evaluate(X, y, classifiers, names, cost_m, weight_bad=5, weight_good=1)
    for name in names:
        final_comparison_data.append({
            "Technique": 'weight_based',
            "Model": name,
            "Mean Loss": results[name]["loss"],
            "Std Loss": results[name]["std"],
            "Mean Acc": results[name]["accuracy"]})
    
    print("\n--- Running Calibration-Based Evaluation ---")
    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(kernel='linear', probability=True),
        GaussianNB()
    ]
    calibrations = ['sigmoid', 'isotonic']
    for cal in calibrations:
        results, fold_matrices = calibration_evaluate(X, y, mapping, classifiers, names, cost_m, calibrate=cal)
        for name in names:
            final_comparison_data.append({
            "Technique": cal,
            "Model": name,
            "Mean Loss": results[name]["loss"],
            "Std Loss": results[name]["std"],
            "Mean Acc": results[name]["accuracy"]})


    final_df = pd.DataFrame(final_comparison_data)
    final_df = final_df.sort_values(by="Mean Loss")

    print("\n" + "="*70)
    print("           FINAL CROSS-VALIDATION PERFORMANCE LEADERBOARD")
    print("="*70)
    print(final_df.to_string(index=False))

    final_df.to_csv("final_leaderboard.csv", index=False)

    final_df['Full_Name'] = final_df['Technique'] + " - " + final_df['Model']
    final_df = final_df.sort_values('Mean Loss')

    plt.figure(figsize=(14, 8))

    bars = plt.bar(
        x=final_df['Full_Name'], 
        height=final_df['Mean Loss'], 
        yerr=final_df['Std Loss'], 
        capsize=5, 
        color=plt.cm.viridis(np.linspace(0, 1, len(final_df))),
        edgecolor='black',
        alpha=0.8
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=90)
    plt.title("Global Cost leaderboard", fontsize=15)
    plt.ylabel("Mean Loss")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/Global_Cost_Leaderboard.png")
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=final_df, 
        x='Model', 
        y='Mean Loss', 
        hue='Technique',
        capsize=.1,
        errorbar=None 
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

    plt.title("Strategy perfomance grouped by Classifier", fontsize=14, pad=20)
    plt.ylabel("Total Expected Loss", fontsize=12)
    plt.xlabel("Classifier", fontsize=12)
    plt.legend(title="Optimization Technique", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/Strategy_Performance_by_Classifier.png")
    plt.close()


if __name__ == "__main__":
    main()