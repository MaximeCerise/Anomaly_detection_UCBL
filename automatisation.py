from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import silhouette_score, calinski_harabasz_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def single_super_model(x_train, x_test, y_train, y_test, model, hparam, n_split, scoring_type):
    name_para = list(hparam.keys())[0]
    rng = hparam[name_para]

    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid=hparam, cv=k_fold, scoring=scoring_type)
    grid.fit(x_train, y_train)
    model_best = grid.best_estimator_

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    ConfusionMatrixDisplay(confusion_matrix(y_test, model_best.predict(x_test))).plot(ax=axs[1])
    axs[1].set_title("Confusion Matrix")

    train_score, val_score = validation_curve(model, x_train, y_train, param_name=name_para, param_range=rng, cv=5)
    axs[0].plot(rng, val_score.mean(axis=1), label='Validation')
    axs[0].plot(rng, train_score.mean(axis=1), label='Train')
    axs[0].set_title("Train Validation Curve")
    axs[0].legend()
    axs[0].set_xlabel("Param Range")
    axs[0].set_ylabel("Score")

    if hasattr(model_best, "predict_proba"):
        y_proba = model_best.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        axs[2].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axs[2].plot([0, 1], [0, 1], 'k--')
        axs[2].set_title("ROC Curve")
        axs[2].set_xlabel("False Positive Rate")
        axs[2].set_ylabel("True Positive Rate")
        axs[2].legend()
    else:
        print("Ce modèle ne supporte pas predict_proba pour la courbe ROC")

    fig.tight_layout()
    plt.show()

    y_pred = model_best.predict(x_test)
    if scoring_type == 'accuracy':
        score = accuracy_score(y_test, y_pred)
        print("Best accuracy score : ", score)
    elif scoring_type == 'precision':
        score = precision_score(y_test, y_pred)
        print("Best precision score: ", score)
    elif scoring_type == 'recall':
        score = recall_score(y_test, y_pred)
        print("Best recall score: ", score)
    elif scoring_type == 'f1':
        score = f1_score(y_test, y_pred)
        print("Best F1 score: ", score)
    else:
        raise ValueError("Unknown scoring type:", scoring_type)
    print(
        "________________________________________________________________________________________________________________________________________________")
    return model_best


def auto_supervised(models, x_train, x_test, y_train, y_test, hparam_list, k_fold, scoring_type):
    best_models = []
    for i in range(len(models)):
        model = models[i]
        hparam = hparam_list[i]
        print("Processing ------ Model : ", model, "-- Hyperparameters : ", hparam, "\n")
        best_models.append(single_super_model(x_train, x_test, y_train, y_test, model, hparam, k_fold, scoring_type))
    return best_models


def auto_supervised(models, x_train, x_test, y_train, y_test, hparam_list, k_fold, scoring_type):
    best_models = []
    for i in range(len(models)):
        model = models[i]
        hparam = hparam_list[i]
        print("Processing ------ Model : ", model, "-- Hyperparameters : ", hparam, "\n")
        best_models.append(single_super_model(x_train, x_test, y_train, y_test, model, hparam, k_fold, scoring_type))
    return best_models


def single_unsuper_model(x_train, x_test, y_test, model, hparam):
    name_para = list(hparam.keys())[0]
    rng = hparam[name_para]

    x_train_part, x_val_part = train_test_split(x_train, test_size=0.2, random_state=42)

    best_score = -np.inf
    best_params = None
    best_model = None
    val_scores = []
    train_scores = []

    for value in rng:
        try:
            params = {name_para: value}
            model.set_params(**params)

            y_train_pred = model.fit_predict(x_train_part)

            train_silhouette = silhouette_score(x_train_part, y_train_pred)
            y_val_pred = model.fit_predict(x_val_part)
            val_silhouette = silhouette_score(x_val_part, y_val_pred)

            train_scores.append(train_silhouette)
            val_scores.append(val_silhouette)

            if val_silhouette > best_score:
                best_score = val_silhouette
                best_params = params
                best_model = model
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            train_scores.append(-np.inf)
            val_scores.append(-np.inf)

    try:
        y_test_pred = best_model.fit_predict(x_test)
        silhouette_avg = silhouette_score(x_test, y_test_pred)
        calinski_harabasz = calinski_harabasz_score(x_test, y_test_pred)

        y_test_pred_binary = (y_test_pred == 1).astype(int)
        y_test_binary = (y_test == 1).astype(int)

        cm = confusion_matrix(y_test_binary, y_test_pred_binary, labels=[0, 1])
    except Exception as e:
        print(f"Error on the test set with best parameters {best_params}: {e}")
        silhouette_avg = -np.inf
        calinski_harabasz = -np.inf
        cm = None

    print(f"Best Hyperparameters: {best_params}")
    print(f"Silhouette Score on Test: {silhouette_avg:.2f}")
    print(f"Calinski-Harabasz Score on Test: {calinski_harabasz:.2f}")
    print("Best model", best_model)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(rng, train_scores, label='Train', marker='o')
    axs[0].plot(rng, val_scores, label='Validation', marker='o')
    axs[0].set_title("Validation Curve")
    axs[0].legend()
    axs[0].set_xlabel(name_para)
    axs[0].set_ylabel("Silhouette Score")

    axs[1].bar(['Silhouette', 'Calinski-Harabasz'], [silhouette_avg, calinski_harabasz], color=['blue', 'orange'])
    axs[1].set_title("Scores on Test Data")
    axs[1].set_ylabel("Score")

    if cm is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomaly", "Normal"])
        disp.plot(cmap='Blues', ax=axs[2], colorbar=False)
        axs[2].set_title("Confusion Matrix")
    else:
        axs[2].set_title("No Confusion Matrix Available")

    fig.tight_layout()
    plt.show()

    return best_model


def auto_unsupervised(models, x_train, x_test, y_test, hparam_list):
    mod = []
    for i in range(len(models)):
        model = models[i]
        hparam = hparam_list[i]
        print(f"Processing Model {i + 1}: {model} -- Hyperparameters: {hparam}")
        mod.append(single_unsuper_model(x_train, x_test, y_test, model, hparam))
    return mod