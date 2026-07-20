import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from catboost import CatBoostClassifier
import optuna


def load_and_engineer(path='data.xlsx'):
    df = pd.read_excel(path)
    df['default'] = (df['Флаг дефолта по кредиту'] == 'Дефолт').astype(int)

    df['monthly_payment'] = df['Сумма кредита'] / df['Срок кредита']
    df['credit_to_income'] = df['Сумма кредита'] / df['Доход клиента'].replace(0, np.nan)
    df['payment_to_income'] = df['monthly_payment'] / df['Доход клиента'].replace(0, np.nan)
    df['income_per_age'] = df['Доход клиента'] / (df['Возраст клиента'] + 1)
    df['credit_x_term'] = df['Сумма кредита'] * df['Срок кредита']
    df['credit_per_month'] = df['Сумма кредита'] / df['Срок кредита']

    df['is_young'] = (df['Возраст клиента'] < 25).astype(int)
    df['is_senior'] = (df['Возраст клиента'] > 55).astype(int)
    df['is_big_loan'] = (df['Сумма кредита'] > df['Сумма кредита'].quantile(0.75)).astype(int)
    df['is_long_term'] = (df['Срок кредита'] > df['Срок кредита'].median()).astype(int)
    df['is_male'] = (df['Пол клиента'] == 'Мужской').astype(int)
    df['has_children'] = (df['Наличие детей у клиента'] == 'Дети есть').astype(int)
    df['is_bank_client'] = (df['Является ли клиентом банка'] == 'Клиент банка').astype(int)
    df['low_education'] = df['Образование клиента'].isin(['Среднее', 'Среднее специальное']).astype(int)
    df['other_region'] = (df['Регион выдачи кредита'] == 'Другие регионы').astype(int)
    df['single'] = (df['Семейное положение'] == 'Никогда не был(а) женат/замужем').astype(int)

    df['income_x_age'] = df['Доход клиента'] * df['Возраст клиента']
    df['loan_x_age'] = df['Сумма кредита'] * df['Возраст клиента']
    df['term_x_age'] = df['Срок кредита'] * df['Возраст клиента']
    df['low_income_big_loan'] = ((df['Доход клиента'] < df['Доход клиента'].quantile(0.25)) &
                                  (df['Сумма кредита'] > df['Сумма кредита'].quantile(0.75))).astype(int)

    df['credit_to_income'] = df['credit_to_income'].fillna(df['credit_to_income'].median())
    df['payment_to_income'] = df['payment_to_income'].fillna(df['payment_to_income'].median())

    return df


def get_features():
    num = ['Месяц выдачи кредита', 'Сумма кредита', 'Срок кредита', 'Возраст клиента', 'Доход клиента',
           'monthly_payment', 'credit_to_income', 'payment_to_income', 'income_per_age', 'credit_x_term',
           'credit_per_month', 'is_young', 'is_senior', 'is_big_loan', 'is_long_term', 'is_male',
           'has_children', 'is_bank_client', 'low_education', 'other_region', 'single',
           'income_x_age', 'loan_x_age', 'term_x_age', 'low_income_big_loan']
    cat = ['Пол клиента', 'Образование клиента', 'Тип товара', 'Наличие детей у клиента',
           'Регион выдачи кредита', 'Семейное положение', 'Оператор связи', 'Является ли клиентом банка']
    return num, cat


def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def apply_business_rules(df):
    df = df.copy()
    monthly_payment = df['Сумма кредита'] / df['Срок кредита']
    credit_to_income = df['Сумма кредита'] / df['Доход клиента'].replace(0, np.nan)
    payment_to_income = monthly_payment / df['Доход клиента'].replace(0, np.nan)

    # АВТООТКАЗ
    auto_reject = (payment_to_income > 0.5) | (credit_to_income > 10)
    df['auto_reject'] = auto_reject.astype(int)

    # МИКРОКРЕДИТЫ
    micro_loan_safe = (df['Сумма кредита'] < 5000) & (payment_to_income < 0.05)
    df['micro_loan_safe'] = micro_loan_safe.astype(int)

    # КОРРЕКТИРОВКИ РИСКА (на основе анализа данных)
    risk_adj = pd.Series(0.0, index=df.index)

    # Правило 1: Молодой клиент + низкий доход = повышенный риск (+10%)
    young_low_income = (df['Возраст клиента'] < 25) & (df['Доход клиента'] < 20000)
    risk_adj += 0.10 * young_low_income.astype(float)

    # Правило 2: Среднее образование = повышенный риск (+8%)
    low_edu = df['Образование клиента'] == 'Среднее'
    risk_adj += 0.08 * low_edu.astype(float)

    # Правило 3: Мобильные телефоны + Среднее образование = ОЧЕНЬ высокий риск (+15%)
    mobile_medium_edu = (df['Тип товара'] == 'Мобильные телефоны') & (df['Образование клиента'] == 'Среднее')
    risk_adj += 0.15 * mobile_medium_edu.astype(float)

    # Правило 4: Ювелирные украшения + Молодой возраст (<25) = очень высокий риск (+12%)
    jewelry_young = (df['Тип товара'] == 'Ювелирные украшения') & (df['Возраст клиента'] < 25)
    risk_adj += 0.12 * jewelry_young.astype(float)

    # Правило 5: Среднее образование + Другие регионы = усиленный риск (+12% вместо 8%)
    edu_region_combo = (df['Образование клиента'] == 'Среднее') & (df['Регион выдачи кредита'] == 'Другие регионы')
    risk_adj += 0.04 * edu_region_combo.astype(float)  # дополнительные +4% поверх базовых +8%

    # Правило 6: Доход 27-38k = повышенный риск (+5%)
    mid_income_risk = (df['Доход клиента'] >= 27000) & (df['Доход клиента'] <= 38000)
    risk_adj += 0.05 * mid_income_risk.astype(float)

    # Правило 7: Короткие кредиты (<=6 мес) = сниженный риск (-5%)
    short_term = df['Срок кредита'] <= 6
    risk_adj -= 0.05 * short_term.astype(float)

    # Правило 8: Высокий доход (>60k) = сниженный риск (-5%)
    high_income = df['Доход клиента'] > 60000
    risk_adj -= 0.05 * high_income.astype(float)

    # Правило 9: СПб регион = сниженный риск (-4%)
    spb_region = df['Регион выдачи кредита'] == 'Санкт-Петербург или ЛО'
    risk_adj -= 0.04 * spb_region.astype(float)

    df['risk_adjustment'] = risk_adj

    return df


def main():
    df = load_and_engineer('data.xlsx')
    df = apply_business_rules(df)
    num_features, cat_features = get_features()
    X = df[num_features + cat_features]
    y = df['default']
    auto_reject = df['auto_reject'].values

    X_train, X_test, y_train, y_test, reject_train, reject_test = train_test_split(
        X, y, auto_reject, test_size=0.2, random_state=42, stratify=y
    )

    print('Подбор гиперпараметров CatBoost...')

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 400, 1200),
            'depth': trial.suggest_int('depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 5.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 254),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'Logloss'
        }
        model = CatBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1,
                                 params={'cat_features': cat_features})
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=80, show_progress_bar=True)

    print(f'Лучший CV PR-AUC: {study.best_value:.4f}')
    print(f'Лучшие параметры: {study.best_params}')

    best_model = CatBoostClassifier(
        **study.best_params,
        random_seed=42,
        verbose=False,
        loss_function='Logloss'
    )
    best_model.fit(X_train, y_train, cat_features=cat_features)

    y_proba_raw = best_model.predict_proba(X_test)[:, 1]

    y_proba = np.where(reject_test == 1, 1.0, y_proba_raw)

    threshold = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    auto_reject_count = int(reject_test.sum())
    auto_reject_pct = float(auto_reject_count / len(y_test) * 100)

    metrics = {
        'Model': 'CatBoost + Business Rules',
        'PR-AUC': float(average_precision_score(y_test, y_proba)),
        'ROC-AUC': float(roc_auc_score(y_test, y_proba)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'F1': float(f1_score(y_test, y_pred, zero_division=0)),
        'Threshold': float(threshold),
        'AutoRejectCount': auto_reject_count,
        'AutoRejectPct': auto_reject_pct,
        'BusinessRules': {
            'payment_to_income_threshold': 0.5,
            'credit_to_income_threshold': 10.0
        },
        'CV_PR_AUC': float(study.best_value),
        'BestParams': study.best_params,
        'ConfusionMatrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print('\nМетрики на тесте:')
    for k, v in metrics.items():
        if k not in ['ConfusionMatrix', 'BestParams', 'BusinessRules']:
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')
    print(f'\nБизнес-правила автоотказа:')
    print(f"  Платёж/доход > {metrics['BusinessRules']['payment_to_income_threshold']} ИЛИ")
    print(f"  Кредит/доход > {metrics['BusinessRules']['credit_to_income_threshold']}")
    print(f"  Автоотказано: {metrics['AutoRejectCount']} клиентов ({metrics['AutoRejectPct']:.1f}%)")
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['Нет дефолта', 'Дефолт']))

    Path('artifacts').mkdir(exist_ok=True)
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'num_features': num_features, 'cat_features': cat_features,
                     'features': num_features + cat_features}, f)

    with open('artifacts/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    feature_config = {
        'features': num_features + cat_features,
        'num_features': num_features,
        'cat_features': cat_features,
        'categories': {col: sorted([str(x) for x in X[col].unique().tolist()]) for col in cat_features},
        'num_stats': {col: {'min': float(X[col].min()), 'max': float(X[col].max()),
                            'mean': float(X[col].mean()), 'std': float(X[col].std())}
                      for col in num_features}
    }
    with open('artifacts/feature_config.json', 'w', encoding='utf-8') as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
