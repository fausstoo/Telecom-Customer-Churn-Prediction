import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, roc_curve, make_scorer, precision_recall_curve
import matplotlib.pyplot as plt

df = pd.read_pickle("../data/raw/customer_churn_dataset-test.pkl")
preprocessor = joblib.load("../artifacts/preprocessor.pkl")
model = joblib.load("../artifacts/best_model.pkl")

cleaned_df = preprocessor.transform(X_true)

df[df['Churn']== 0]


y_scores = model.predict_proba(cleaned_df)[:, 1]

y_true = df.iloc[:,-1]
X_true = df.iloc[:,:-1]

y_pred = model.predict(cleaned_df)


def plot_precision_recall_curve(y_true, y_scores, save_path=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Calculate Precision-Recall AUC
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.legend(loc='lower left')
    
    # Save figure
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        precision_recall_curve_save_path = os.path.join(save_path, "precision_recall_curve.png")
        plt.savefig(precision_recall_curve_save_path, bbox_inches='tight', dpi=300)
    print(f"Saved at {save_path}")

    return plt


plot_precision_recall_curve(y_true, y_scores)


def plot_roc_auc_curve(y_true, y_scores, save_path=None):
    # ROC AUC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save the plot
    if save_path:
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)  
        roc_auc_curve_save_path = os.path.join(save_path, "roc_auc_curve.png")
        plt.savefig(roc_auc_curve_save_path, bbox_inches='tight', dpi=300)
    print(f"Saved at {save_path}")
    
    return plt

plot_roc_auc_curve(y_true, y_scores)