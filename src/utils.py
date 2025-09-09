# utils.py
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def visualize_actual_vs_predicted(
    all_predictions,
    all_values,
    all_aggregate,
    column_name,
    devices=None,
    device_index=None,
    smooth=None,
    start=None,
    end=None,
    ax=None,
    save_path=None,
    show=True,
    compute_metrics=True
):
    def _tolist(batch):
        try:
            return batch.tolist()
        except AttributeError:
            return batch

    preds = [row for sub in all_predictions for row in _tolist(sub)]
    trues = [row for sub in all_values for row in _tolist(sub)]
    aggs = [row[0] for sub in all_aggregate for row in _tolist(sub)] if all_aggregate else []

    if not preds or not trues:
        print("No predictions or ground-truth values to plot.")
        return
    if not column_name:
        print("No device columns (column_name) provided.")
        return

    df_pred = pd.DataFrame(preds, columns=column_name) if isinstance(preds[0], (list, tuple)) else pd.DataFrame(preds)
    df_true = pd.DataFrame(trues, columns=column_name) if isinstance(trues[0], (list, tuple)) else pd.DataFrame(trues)

    n = min(len(df_pred), len(df_true))
    if n == 0:
        print("Nothing to plot after length alignment.")
        return
    df_pred = df_pred.iloc[:n].reset_index(drop=True)
    df_true = df_true.iloc[:n].reset_index(drop=True)

    if smooth is not None:
        try:
            window = int(smooth)
            if window > 1:
                df_pred = df_pred.rolling(window=window, min_periods=1).mean()
                df_true = df_true.rolling(window=window, min_periods=1).mean()
        except Exception:
            pass

    s = 0 if start is None else max(0, int(start))
    e = n if end is None else min(n, int(end))
    if s < e:
        df_pred = df_pred.iloc[s:e]
        df_true = df_true.iloc[s:e]
    else:
        print("Invalid slice range; nothing to plot.")
        return

    indices = []
    if devices is not None:
        if isinstance(devices, str):
            devices = [devices]
        indices = [column_name.index(name) for name in devices if name in column_name]
    elif device_index is not None:
        if isinstance(device_index, int):
            indices = [device_index]
        else:
            indices = list(device_index)
    else:
        indices = [0]

    indices = [i for i in indices if 0 <= i < len(column_name)]
    if not indices:
        print("No valid device indices to plot.")
        return

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
        created_fig = True

    metrics_summary = {}
    for i in indices:
        name = column_name[i]
        y_true = df_true[name] if name in df_true.columns else df_true.iloc[:, i]
        y_pred = df_pred[name] if name in df_pred.columns else df_pred.iloc[:, i]

        ax.plot(y_true.values, label=f'Actual - {name}')
        ax.plot(y_pred.values, label=f'Predicted - {name}', alpha=0.85)

        if compute_metrics:
            yt = y_true.values
            yp = y_pred.values
            mae = float(np.mean(np.abs(yp - yt)))
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) if len(yt) > 1 else 0.0
            # Robust R2 to avoid NaN when target is constant or too short
            var = float(np.var(yt))
            if len(yt) < 2 or var < 1e-12:
                r2 = 1.0 if np.allclose(yt, yp) else 0.0
            else:
                r2 = float(1 - ss_res / ss_tot)
            metrics_summary[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    ax.set_title('Predicted vs Actual')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(loc='best')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if created_fig and show:
        plt.show()
    elif created_fig and not show:
        plt.close()

    if compute_metrics and metrics_summary:
        for dev, m in metrics_summary.items():
            print(f"{dev}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")


def save_result(all_predictions, all_values, all_aggregate, test_data, column_name):

    all_preds = [item for sublist in all_predictions for item in sublist.tolist()]  # Flatten all_predictions list
    true_value = [item for sublist in all_values for item in sublist.tolist()]
    all_aggs = [item[0] for sublist in all_aggregate for item in sublist.tolist()]
    df = pd.DataFrame(all_preds, columns=column_name)
    df['Aggregate'] = all_aggs

    # Select the last six columns of test_data (original device status)
    selected_columns = test_data.iloc[:, -6:]
    selected_columns = selected_columns.reset_index(drop=True)
    # Merge data horizontally
    result = pd.concat([df, selected_columns], axis=1)
    result.to_csv('output-test.csv', index=False, encoding='utf-8-sig')


def get_scheduler(optimizer, scheduler_config):
    if scheduler_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=scheduler_config['step_size'],
                                                    gamma=scheduler_config['gamma'])
    elif scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=scheduler_config['factor'],
                                                              patience=scheduler_config['patience'])
    # Add conditions for other scheduler types...
    else:
        raise NotImplementedError("The specified scheduler type is not implemented")

    return scheduler


def classification(result, column_name):

    data = pd.read_csv("../data/data.csv")
    data_test = result.iloc[:, 2:]
    # print(data_test)
    df = pd.DataFrame(data)
    df_test = pd.DataFrame(data_test)
    # Build a model for each device
    models = {}
    devices = column_name
    predicted_status = pd.DataFrame()

    for device in devices:
        X = df[device].values.reshape(-1, 1)  # load data
        y = df[device + '_State']  # device status

        # Split dataset into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X
        y_train = y
        # Train using a decision tree classifier
        # clf = GradientBoostingClassifier(n_estimators=100)
        clf = DecisionTreeClassifier()
        # clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        # Store the model
        models[device] = clf

        X_test = df_test[device].values.reshape(-1, 1)
        y_pred = clf.predict(X_test)
        # Add predictions to predicted_status DataFrame
        predicted_status[device + '_State'] = y_pred

    return predicted_status

def classification(result, column_name, data_csv_path="../data/data.csv", pred_suffix="_pred", return_metrics=False):

    df_raw = pd.read_csv(data_csv_path)
    out = pd.DataFrame(index=result.index)
    metrics = {}

    for device in column_name:
        # Training: power -> state (store as strings, strip spaces)
        X_train = df_raw[[device]].values
        y_train_raw = df_raw[f"{device}_State"].astype(str).str.strip()
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Prediction: prefer *_pred column (fallback to original column if not available)
        pred_col = f"{device}{pred_suffix}" if f"{device}{pred_suffix}" in result.columns else device
        x_test = result[[pred_col]].values
        y_pred_enc = clf.predict(x_test)
        # Store string states
        out[f"{device}_State_Pred"] = le.inverse_transform(y_pred_enc)

        # Align ground truth (by original index)
        if f"{device}_State" in df_raw.columns:
            valid_idx = [i for i in result.index if 0 <= i < len(df_raw)]
            true_series = pd.Series(index=result.index, dtype="object")
            if valid_idx:
                true_vals = df_raw.loc[valid_idx, f"{device}_State"].astype(str).str.strip().values
                true_series.loc[valid_idx] = true_vals
            out[f"{device}_State_True"] = true_series

            # Compute metrics (only on rows with ground truth)
            valid_mask = true_series.notna()
            if valid_mask.any():
                y_true_enc = le.transform(true_series[valid_mask])
                y_pred_enc_aligned = le.transform(out.loc[valid_mask, f"{device}_State_Pred"])

                acc = accuracy_score(y_true_enc, y_pred_enc_aligned)
                # For binary classification: treat "ON" as positive; otherwise use macro average
                if len(le.classes_) == 2 and ("ON" in le.classes_):
                    pos_label_enc = int(np.where(le.classes_ == "ON")[0][0])
                    pr, rc, f1, _ = precision_recall_fscore_support(
                        y_true_enc, y_pred_enc_aligned, average="binary",
                        pos_label=pos_label_enc, zero_division=0
                    )
                else:
                    pr, rc, f1, _ = precision_recall_fscore_support(
                        y_true_enc, y_pred_enc_aligned, average="macro", zero_division=0
                    )

                metrics[device] = {
                    "accuracy": float(acc),
                    "precision": float(pr),
                    "recall": float(rc),
                    "f1": float(f1),
                    "n": int(valid_mask.sum())
                }

    if return_metrics:
        return out, metrics
    return out