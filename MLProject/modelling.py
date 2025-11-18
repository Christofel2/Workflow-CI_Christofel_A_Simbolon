import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
import json
import dagshub
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss,
                             matthews_corrcoef, confusion_matrix,
                             average_precision_score, ConfusionMatrixDisplay)

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment("Loan Approval - Tuning ")

# # --- KONFIGURASI DagsHub ---
# DAGSHUB_REPO_OWNER = "Christofel2"
# DAGSHUB_REPO_NAME = "Submission-MSML"

# # Inisialisasi DagsHub
# dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

# mlflow.set_experiment("Loan Approval- Tuning ") 


def train_with_tuning():
    # Load Dataset 
    try:
        data = pd.read_csv("preprossed_dataset/loan_approved_processed.csv")
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        try:
            data = pd.read_csv("loan_approved_processed.csv")
            print("Dataset loaded from local directory.")
        except:
            try:
                data = pd.read_csv("../preprocessing/loan_approved_processed.csv")
            except:
                print("Dataset tidak ditemukan!")
                return

    target_col = "loan_approved"
    if target_col not in data.columns:
        print(f"Kolom target '{target_col}' tidak ditemukan!")
        return
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    xgb = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1
    )
    
    print(" Grid Search Start" )

    with mlflow.start_run(run_name="XGBoost_GridSearch_Tuning"):

        print("Logging dataset metadata...")
        train_df = X_train.copy()
        train_df[target_col] = y_train
        
        dataset = mlflow.data.from_pandas(
            train_df, 
            targets=target_col, 
            name="Loan_Processed_Train_Tuning"
        )
        mlflow.log_input(dataset, context="training")

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best Param: {best_params}")


        #LOGGING PARAMETER
        for p, v in best_params.items():
            mlflow.log_param(f"best_{p}", v)


        # METRIK
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')
        f1   = f1_score(y_test, y_pred, average='weighted')
        roc  = roc_auc_score(y_test, y_proba)
        ll   = log_loss(y_test, y_proba)
        mcc    = matthews_corrcoef(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_proba)


        # LOG SEMUA METRIK
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("log_loss", ll)
        mlflow.log_metric("matthews_corrcoef", mcc)
        mlflow.log_metric("pr_auc", pr_auc)

 
        #LOG MODEL
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model_tuning",
            input_example=X_train.iloc[:5]
        )
        
        #ARTEFAK TAMBAHAN
        # JSON 
        with open("best_params.json", "w") as f:
            json.dump(best_params, f)
        mlflow.log_artifact("best_params.json")

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        print(f" DONE | Accuracy = {acc:.4f} | ROC AUC = {roc:.4f}")
        print("Artefak tersimpan di MLflow UI")

if __name__ == "__main__":
    train_with_tuning()
