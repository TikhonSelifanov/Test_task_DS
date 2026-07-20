import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_psi(expected, actual, bins=10):
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0 or e_perc == 0:
            return 0
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents)))
    return psi_value


def main():
    df = pd.read_excel('data.xlsx')
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

    with open('artifacts/model.pkl', 'rb') as f:
        artifact = pickle.load(f)
    model = artifact['model']
    num_features = artifact['num_features']
    cat_features = artifact['cat_features']

    X = df[num_features + cat_features]
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df['score'] = model.predict_proba(X)[:, 1]

    df['period'] = 'train'
    df.loc[X_test.index, 'period'] = 'test'

    monthly = df.groupby('Месяц выдачи кредита').apply(
        lambda x: pd.Series({
            'count': len(x),
            'default_rate': x['default'].mean(),
            'mean_score': x['score'].mean(),
            'pr_auc': average_precision_score(x['default'], x['score']) if x['default'].sum() > 0 else np.nan,
            'roc_auc': roc_auc_score(x['default'], x['score']) if x['default'].nunique() > 1 else np.nan
        })
    ).reset_index()

    train_score = df.loc[X_train.index, 'score']
    test_score = df.loc[X_test.index, 'score']
    score_psi = calculate_psi(train_score.values, test_score.values)

    psi_results = []
    key_features = ['Сумма кредита', 'Срок кредита', 'Возраст клиента', 'Доход клиента']
    for col in key_features:
        psi = calculate_psi(X_train[col].values, X_test[col].values)
        psi_results.append({'feature': col, 'psi': psi})
    psi_df = pd.DataFrame(psi_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(monthly['Месяц выдачи кредита'], monthly['pr_auc'], marker='o', label='PR-AUC')
    axes[0, 0].plot(monthly['Месяц выдачи кредита'], monthly['roc_auc'], marker='s', label='ROC-AUC')
    axes[0, 0].set_title('Качество модели по месяцам')
    axes[0, 0].set_xlabel('Месяц')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].bar(monthly['Месяц выдачи кредита'], monthly['default_rate'], color='#E45756')
    axes[0, 1].set_title('Доля дефолтов по месяцам')
    axes[0, 1].set_xlabel('Месяц')
    axes[0, 1].set_ylabel('Доля дефолтов')

    axes[1, 0].plot(monthly['Месяц выдачи кредита'], monthly['mean_score'], marker='o', color='#72B7B2')
    axes[1, 0].set_title('Средний скор риска по месяцам')
    axes[1, 0].set_xlabel('Месяц')
    axes[1, 0].set_ylabel('Средний скор')
    axes[1, 0].grid(True)

    colors = ['#4C78A8' if p < 0.1 else '#F58518' if p < 0.25 else '#E45756' for p in psi_df['psi']]
    axes[1, 1].barh(psi_df['feature'], psi_df['psi'], color=colors)
    axes[1, 1].axvline(0.1, color='green', linestyle='--', label='PSI < 0.1')
    axes[1, 1].axvline(0.25, color='red', linestyle='--', label='PSI < 0.25')
    axes[1, 1].set_title('PSI ключевых признаков (train vs test)')
    axes[1, 1].set_xlabel('PSI')
    axes[1, 1].legend()

    plt.tight_layout()
    Path('artifacts').mkdir(exist_ok=True)
    plt.savefig('artifacts/stability.png', dpi=150, bbox_inches='tight')
    print('График стабильности сохранен: artifacts/stability.png')

    report = {
        'monthly_performance': monthly.round(4).to_dict(orient='records'),
        'score_psi': float(score_psi),
        'feature_psi': psi_df.round(4).to_dict(orient='records'),
        'stability_assessment': 'стабильна' if score_psi < 0.1 else 'средняя нестабильность' if score_psi < 0.25 else 'нестабильна'
    }
    with open('artifacts/stability_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'Score PSI: {score_psi:.4f}')
    print(psi_df.round(4))


if __name__ == '__main__':
    main()
