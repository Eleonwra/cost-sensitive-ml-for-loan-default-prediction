# Cost-Sensitive-ML-for-Loan-Default-Prediction-
In credit scoring, the cost of failing to detect a high-risk borrower is significantly higher than the cost of rejecting a good borrower. Standard accuracy-driven models ignore this, leading to suboptimal financial decisions.
This project provides a comparative analysis of three distinct techniques to solve this asymmetry on the German Credit Dataset:
1.	Data-Level: Hybrid Resampling.
2.	Algorithm-Level: Cost-Weighting.
3.	Post-Processing: Bayes Risk Minimization (with Probability Calibration).

## Installation & Usage

'''bash
#Clone the repo
git clone https://github.com/your-username/credit-guard.git

#Install dependencies
pip install -r requirements.txt

#Run the full evaluation pipeline
python main.py
'''

## Evaluation 
As the primary objective was to compare 3 different cost-sensitive techniques rather than optimizing individual model performance, a standard 5-Fold Stratified Cross-Validation was used.

## Metrics
•	Accuracy
•	Total Financial Cost. This is calculated using a Hadamard product of the model's Confusion Matrix and the predefined Cost Matrix:
$$Total\ Cost = \sum (Confusion\ Matrix \odot Cost\ Matrix)$$
By multiplying these matrices element-wise, the model is billed for the specific financial impact of its errors. Summing these results provides the final metric used for ranking.

## Optimization Strategies
Three methodologies were evaluated to align model behavior with the 1:5 asymmetric cost ratio (False Positive vs. False Negative).

### Data- Level: Hybrid Resampling
•	Logic: The majority class ("Good") is undersampled by 50%, followed by oversampling the minority class ("Bad") until the 5:1 cost-analogous ratio is achieved.
•	Example: In a fold with 500 Good/100 Bad samples, the resulting training set is 250 Good/1250 Bad. This forces the model to prioritize the expensive class as the statistical majority.

### Algorithm-Level: Cost-Weighting
•	Logic: Financial penalties are injected directly into the model's internal optimization logic. Class_weight of 5 is assigned to "Bad" samples, scaling the error contribution during training.
•	Example: In a Random Forest, a misclassified "Bad" applicant increases Gini Impurity five times more than a misclassified "Good" applicant. Tree nodes split specifically to isolate high-cost instances.

### Post-Processing: Bayes Risk Minimization
•	Logic: Instead of selecting the most likely class, the model selects the action $i$ that minimizes the Expected Bayes Cost:
$$R(a_i|x) = \sum_{j} P(\text{class}_j|x) \cdot C(\text{pred}_i, \text{true}_j)$$
This ensures that the final prediction is not just the most "probable" outcome, but the one that carries the lowest financial risk for the institution.
•	Example: Consider a borrower with a 25% predicted probability of default.
o	Granting the loan carries an expected cost of 1.25 (0.25 x 5 penalty).
o	Rejecting the loan carries an expected cost of 0.75 (0.75 x 1 penalty).
o	Decision: The system rejects the application because the risk-adjusted cost of rejection is lower than the risk-adjusted cost of granting the loan.
•	Calibration: The validity of the Bayes Risk calculation depends entirely on the probabilistic integrity of the model. Many classifiers produce biased probability estimates (overconfident or underconfident). To ensure these values represent true empirical frequencies, Probability Calibration (via Isotonic Regression or Platt Scaling/Sigmoid) is applied. This step transforms raw model scores into reliable risk estimates.


## Results & Visual Analysis
•	Cost Leaderboard: A bar chart comparing the total financial loss across strategies.

## Key Findings
•	Cost vs. Accuracy: Optimizing for the lowest financial cost often requires sacrificing global accuracy to ensure high-cost defaults are caught.
•	Optimization Success: All three cost-sensitive methodologies consistently outperformed the baseline by prioritizing the 5:1 penalty ratio during the decision-making process.
