import glob
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Tuple, List, Dict

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def parse_and_train_credit_card_dataset(seed=21) -> pd.DataFrame:
    # Load your ARFF file
    path = 'data/default of credit card clients.arff'  # Replace with your ARFF file path
    data, meta = arff.loadarff(path)
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    target = 'y'
    df[target] = df[target].astype(int)
    
    columns = [col for col in df.columns if col not in ['id', target]]
    
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    train, valid = train_test_split(train, test_size=0.1, random_state=seed)
    
    scale_columns = [f'x{i}' for i in range(12, 24)]
    
    scaler = StandardScaler()
    scaler.fit(train[scale_columns])
    
    train[scale_columns] = scaler.transform(train[scale_columns])
    valid[scale_columns] = scaler.transform(valid[scale_columns])
    test[scale_columns] = scaler.transform(test[scale_columns])
    
    rf_clf = RandomForestClassifier(class_weight='balanced', random_state=seed)
    xgb_clf = xgb.XGBClassifier(random_state=seed)
    
    print('training rf')
    rf_clf.fit(train[columns], train[target])
    print('training xgb')
    xgb_clf.fit(train[columns], train[target])
    
    rf_proba_valid = rf_clf.predict_proba(valid[columns])[:, 1]
    xgb_proba_valid = xgb_clf.predict_proba(valid[columns])[:, 1]
    
    rf_proba_test = rf_clf.predict_proba(test[columns])[:, 1]
    xgb_proba_test = xgb_clf.predict_proba(test[columns])[:, 1]
    
    rf_pred_test = rf_clf.predict(test[columns])
    xgb_pred_test = xgb_clf.predict(test[columns])
    
    print('rf acc', accuracy_score(test[target], rf_pred_test))
    print('xgb acc', accuracy_score(test[target], xgb_pred_test))

    print('rf auc', roc_auc_score(test[target], rf_pred_test))
    print('xgb auc', roc_auc_score(test[target], xgb_pred_test))
    
    print('rf ks', np_ks(test[target], rf_pred_test))
    print('xgb ks', np_ks(test[target], xgb_pred_test))

    num_grade = 10

    _, bins = pd.qcut(rf_proba_valid, num_grade, labels=range(1, num_grade+1), retbins=True)
    bins[0] = -np.inf
    bins[-1] = np.inf
    rf_test_grade = pd.cut(rf_proba_test, bins=bins, labels=range(1, num_grade+1))

    _, bins = pd.qcut(xgb_proba_valid, num_grade, labels=range(1, num_grade+1), retbins=True)
    bins[0] = -np.inf
    bins[-1] = np.inf
    xgb_test_grade = pd.cut(xgb_proba_test, bins=bins, labels=range(1, num_grade+1))
    
    tmp = pd.DataFrame()
    tmp['rf_proba'] = rf_proba_test
    tmp['rf_grade'] = rf_test_grade
    tmp['rf_grade'] = tmp['rf_grade'].astype(int)

    tmp['xgb_proba'] = xgb_proba_test
    tmp['xgb_grade'] = xgb_test_grade
    tmp['xgb_grade'] = tmp['xgb_grade'].astype(int)
    
    tmp[target] = test[target].values

    return tmp


def np_ks(label, proba) -> float:
    label_rev = 1 - label
    proba_rev = 1 - proba

    data = np.dstack((proba_rev, label, label_rev)).reshape(-1, 3)
    data = data[np.argsort(proba_rev), :]
    defaults_cum_rate = data[:, 1].cumsum() / label.sum()
    nondefaults_cum_rate = data[:, 2].cumsum() / label_rev.sum()
    max_ks_np = (defaults_cum_rate - nondefaults_cum_rate).max()

    return max_ks_np.round(4)


def sample_from_beta(mean, alpha=0.1) -> float:
    beta = (alpha - mean * alpha) / mean
    return np.random.beta(alpha, beta)


def apply_bank_proba(merged_df, beta_mean=0.1, alpha=0.1, css_type=None) -> pd.DataFrame:
    if css_type is None:
        css_type = np.random.choice(['rf', 'xgb'])
    tmp = merged_df.groupby(f'{css_type}_grade')['y'].mean().reset_index()
    tmp['gamma'] = tmp['y'].apply(lambda x: sample_from_beta(beta_mean, alpha=alpha))
    tmp['interest'] = round(tmp['y'] + tmp['gamma'] * tmp['y'], 4)
    tmp['bank_grade'] = tmp[f'{css_type}_grade'].astype(int)
    tmp['default_proba'] = tmp['y']

    tmp = tmp[[f'{css_type}_grade', 'interest', 'bank_grade', 'default_proba']]
    return tmp


def sample_amount_from_normal(mean_amount, std_amount) -> int:
    amount = np.random.normal(mean_amount, scale=std_amount)
    amount = max(500, min(5000, amount))
    return int(amount)


def get_random_bank_df(df, seed=None, mean_amount=2700, std_amount=500, alpha=0.5, force_css_type=None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    merged_df = df.copy()
    for i in range(10):
        bank_df = apply_bank_proba(df, alpha=alpha, css_type=force_css_type)
        bank_df = bank_df.rename(columns={'interest': f'p{i}_INTEREST', 'bank_grade': f'p{i}_GRADE',
                                          'default_proba': f'p{i}_DEFAULT_PROBA'})
        key = 'xgb_grade'
        if 'rf_grade' in bank_df.columns:
            key = 'rf_grade'
        merged_df = pd.merge(merged_df, bank_df, on=key, how='left')
    merged_df["AMOUNT"] = [sample_amount_from_normal(mean_amount, std_amount) for _ in range(len(merged_df))]
    xgb_default_list = merged_df.groupby('xgb_grade')['y'].mean().values
    xgb_default_list = (xgb_default_list[xgb_default_list < 0.2]).tolist()

    rf_default_list = merged_df.groupby('rf_grade')['y'].mean().values
    rf_default_list = (rf_default_list[rf_default_list < 0.2]).tolist()

    xgb_default_list = [round(x, 4) for x in xgb_default_list]
    rf_default_list = [round(x, 4) for x in rf_default_list]

    min_grade = min(len(xgb_default_list), len(rf_default_list))
    merged_df = merged_df[(merged_df['rf_grade'] <= min_grade) & (merged_df['xgb_grade'] <= min_grade)]
    
    return merged_df
    