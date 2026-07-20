import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
import shap


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


def main():
    df = load_and_engineer('data.xlsx')

    with open('artifacts/model.pkl', 'rb') as f:
        artifact = pickle.load(f)
    model = artifact['model']
    num_features = artifact['num_features']
    cat_features = artifact['cat_features']

    X = df[num_features + cat_features]
    y = df['default']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title('SHAP: влияние признаков на прогноз дефолта')
    plt.tight_layout()
    Path('artifacts').mkdir(exist_ok=True)
    plt.savefig('artifacts/shap_summary.png', dpi=150, bbox_inches='tight')
    print('SHAP summary сохранен: artifacts/shap_summary.png')

    importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'].head(15)[::-1], importance['importance'].head(15)[::-1], color='#72B7B2')
    plt.title('Топ-15 важных признаков (SHAP)')
    plt.xlabel('Среднее абсолютное влияние')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=150, bbox_inches='tight')
    print('Feature importance сохранен: artifacts/feature_importance.png')

    with open('artifacts/shap_importance.json', 'w', encoding='utf-8') as f:
        json.dump(importance.head(15).to_dict(orient='records'), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
