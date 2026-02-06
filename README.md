# Cost Sensitive ML for Loan Default Prediction
In credit scoring, the cost of failing to detect a high-risk borrower is significantly higher than the cost of rejecting a good borrower. Standard accuracy-driven models ignore this, leading to suboptimal financial decisions.
This project provides a comparative analysis of three distinct techniques to solve this asymmetry on the [German Credit Dataset](http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data):
1.	**Data-Level: Hybrid Resampling.**
2.	**Algorithm-Level: Cost-Weighting.**
3.	**Post-Processing: Bayes Risk Minimization (with Probability Calibration).**

## Installation & Usage

```bash
#Clone the repo
git clone https://github.com/Eleonwra/Cost-Sensitive-ML-for-Loan-Default-Prediction-.git

#Install dependencies
pip install -r requirements.txt

#Run the full evaluation pipeline
python main.py
```
## **Preprossesing**
- **Feature Engineering**: Although Random Forest is conceptually capable of handling categorical variables, One-Hot Encoding was applied to ensure architectural consistency across the three model families. This transformation was technically required for the Linear SVC and Gaussian Naive Bayes implementations and facilitated a standardized feature space for the comparative cost-analysis.

## **Model Selection**
### **Generative Family: Gaussian Naive Bayes**
- **Logic**: It uses Bayes' Theorem to model the underlying distribution of each class, assuming features are independent and follow a Gaussian curve.

- **Role**: Serves as a test for probabilistic integrity. Because it "naively" assumes features are independent, it suffers from the Double-Counting Effect: redundant information (like highly correlated financial variables) is treated as separate evidence. This results in "pushed" probabilities—the model becomes overconfident, hallucinating 0% or 100% risk when the reality is much more nuanced.

### **Geometric Family: Linear Support Vector Classifier**

- **Logic**: A discriminative model that finds the optimal linear hyperplane to maximize the margin between "Good" and "Bad" borrowers.

- **Role**: Establishes a high-dimensional baseline. It demonstrates how a non-probabilistic, margin-based model can be adapted for cost-sensitive tasks.

- **Probability Mapping**: As the SVC objective function is non-probabilistic, optimizing for the geometric margin rather than class likelihood—probabilities were derived via Platt Scaling. This involves fitting a logistic regression model to the SVC’s uncalibrated decision scores (the signed distances from the hyperplane) to map them into a $[0, 1]$ probability range.

### **Ensemble Family: Random Forest**

- **Logic**: A non-linear approach that aggregates the predictions of multiple decision trees to reduce variance and capture complex feature interactions.

- **Role**: Represents modern non-linear benchmarks. It generates probabilities through ensemble voting. By providing a non-linear contrast to the Linear SVC and Gaussian Naive Bayes, the Random Forest validates whether cost-sensitive strategies remain effective when applied to high-capacity, algorithmic-level models.

## Optimization Strategies
Three methodologies were evaluated to align model behavior with the 1:5 asymmetric cost ratio (False Positive vs. False Negative).

### Data- Level: Hybrid Resampling
-	**Logic**: The majority class ("Good") is undersampled by 50%, followed by oversampling the minority class ("Bad") until the 1:5 cost-analogous ratio is achieved.
-	**Example**: In a fold with 500 Good/100 Bad samples, the resulting training set is 250 Good/1250 Bad. This forces the model to prioritize the expensive class as the statistical majority.

### Algorithm-Level: Cost-Weighting
-	**Logic**: Financial penalties are injected directly into the model's internal optimization logic. Class_weight of 5 is assigned to "Bad" samples, scaling the error contribution during training.
-	**Example**: In a Random Forest, a misclassified "Bad" applicant increases Gini Impurity five times more than a misclassified "Good" applicant. Tree nodes split specifically to isolate high-cost instances.

### Post-Processing: Bayes Risk Minimization
-	**Logic**: Instead of selecting the most likely class, the model selects the action $i$ that minimizes the Expected Bayes Cost:
  <br/>
  
  $$ a^* = \arg\min_{a_i} \sum_{j} P(\text{class}_j|x) \cdot C(\text{pred}_i, \text{true}_j)$$
  
  <br/>
  This ensures that the final prediction is not just the most "probable" outcome, but the one that carries the lowest financial risk for the institution.


-	**Example**: Consider a borrower with a 25% predicted probability of default. 
1.	Granting the loan carries an expected cost of 1.25 (0.25 x 5 penalty).
2.	Rejecting the loan carries an expected cost of 0.75 (0.75 x 1 penalty).
3.	Decision: The system rejects the application because the risk-adjusted cost of rejection is lower than the risk-adjusted cost of granting the loan.
   
The calculation is based on the Cost Matrix of the German dataset used in this project, where a penalty of 5 is assigned to False Negatives (accepting a bad borrower) and 1 to False Positives (rejecting a good borrower).

-	**Calibration**: The validity of the Bayes Risk calculation depends entirely on the probabilistic integrity of the model. Many classifiers produce biased probability estimates (overconfident or underconfident). To ensure these values represent true empirical frequencies, Probability Calibration (via Isotonic Regression and Platt Scaling/Sigmoid) is applied. 

## Evaluation 
As the primary objective was to compare 3 different cost-sensitive techniques rather than optimizing individual model performance, a standard 5-Fold Stratified Cross-Validation was used.

## Metrics
-	**Total Financial Cost**: The primary metric used for ranking. This is calculated as the **mean cost across all 5 folds** to ensure the result is statistically robust. The cost for each fold is derived using the Hadamard product of the Confusion Matrix and the predefined Cost Matrix:

<br/>

$$Total\ Cost = \sum (Confusion\ Matrix \odot Cost\ Matrix)$$

<br/>

-	**Accuracy**: Used as a secondary diagnostic to monitor the trade-off between general predictive power and risk mitigation.

## Results & Visual Analysis
-	**Global Cost Leaderboard**: A bar chart with error bars comparing the total financial loss across strategies and classifiers.
-	**Strategy Performance by Classifier**: A grouped bar chart displaying the financial performance of each specific algorithm.

## Key Findings
-	**Cost vs. Accuracy**: Optimizing for the lowest financial cost often requires sacrificing global accuracy to ensure high-cost defaults are caught.
-	**Optimization Success**: All three cost-sensitive methodologies consistently outperformed the baseline by prioritizing the 5:1 penalty ratio during the decision-making process. However, this success was **not universal** as Random Forest remained the outlier where the weight was neutralized by the model's internal voting logic. Because Random Forest relies on majority voting, a leaf node with 10 "Good" samples still outweighs 1 "Bad" sample ($10 > 5$).
-	**Best method**: Bayes Risk Minimization performed best overall, but showed diminished gains with Naive Bayes. This is attributed to the "Naive" independence assumption, which produces biased probability estimates (The "Double-Counting" Effect) that undermine the cost-minimization logic.
-	**Callibration**: As expected, Sigmoid was the best fit for SVC because its output naturally follows a perfect S-shape, and the fixed formula provides the stability needed to prevent overfitting. Random Forest isn't as stable with Sigmoid because the model's complex voting logic is too "advanced" for a simple formula. If the Random Forest’s voting bias doesn't happen to look like a perfect $S$, the Sigmoid will try to "force" it into that shape, often resulting in probabilities that are still inaccurate. It requires the flexibility of Isotonic to handle its non-linearities, and it only stays stable because the ensemble architecture acts as a safety net that prevents the Isotonic staircase from becoming too erratic.

