import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title='Кредитный скоринг', layout='wide')


@st.cache_resource
def load_model():
    with open('artifacts/model.pkl', 'rb') as f:
        artifact = pickle.load(f)
    with open('artifacts/feature_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open('artifacts/metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return artifact, config, metrics


def load_and_engineer(input_dict):
    df = pd.DataFrame([input_dict])

    monthly_payment = df['Сумма кредита'] / df['Срок кредита']
    credit_to_income = df['Сумма кредита'] / df['Доход клиента'].replace(0, np.nan)
    payment_to_income = monthly_payment / df['Доход клиента'].replace(0, np.nan)

    df['monthly_payment'] = monthly_payment
    df['credit_to_income'] = credit_to_income
    df['payment_to_income'] = payment_to_income
    df['income_per_age'] = df['Доход клиента'] / (df['Возраст клиента'] + 1)
    df['credit_x_term'] = df['Сумма кредита'] * df['Срок кредита']
    df['credit_per_month'] = df['Сумма кредита'] / df['Срок кредита']
    df['is_young'] = (df['Возраст клиента'] < 25).astype(int)
    df['is_senior'] = (df['Возраст клиента'] > 55).astype(int)

    df['is_male'] = (df['Пол клиента'] == 'Мужской').astype(int)
    df['has_children'] = (df['Наличие детей у клиента'] == 'Дети есть').astype(int)
    df['is_bank_client'] = (df['Является ли клиентом банка'] == 'Клиент банка').astype(int)
    df['low_education'] = df['Образование клиента'].isin(['Среднее', 'Среднее специальное']).astype(int)
    df['other_region'] = (df['Регион выдачи кредита'] == 'Другие регионы').astype(int)
    df['single'] = (df['Семейное положение'] == 'Никогда не был(а) женат/замужем').astype(int)

    df['is_big_loan'] = (df['Сумма кредита'] > 34000).astype(int)
    df['is_long_term'] = (df['Срок кредита'] > 12).astype(int)

    df['income_x_age'] = df['Доход клиента'] * df['Возраст клиента']
    df['loan_x_age'] = df['Сумма кредита'] * df['Возраст клиента']
    df['term_x_age'] = df['Срок кредита'] * df['Возраст клиента']
    df['low_income_big_loan'] = ((df['Доход клиента'] < 21000) & (df['Сумма кредита'] > 34000)).astype(int)

    df['credit_to_income'] = df['credit_to_income'].fillna(df['credit_to_income'].median())
    df['payment_to_income'] = df['payment_to_income'].fillna(df['payment_to_income'].median())

    return df


def check_business_rules(input_dict):
    monthly_payment = input_dict['Сумма кредита'] / input_dict['Срок кредита']
    credit_to_income = input_dict['Сумма кредита'] / input_dict['Доход клиента']
    payment_to_income = monthly_payment / input_dict['Доход клиента']

    # Автоотказ
    auto_reject = (payment_to_income > 0.5) or (credit_to_income > 10)
    reject_reasons = []
    if payment_to_income > 0.5:
        reject_reasons.append(f'Ежемесячный платёж ({monthly_payment:,.0f} руб.) превышает 50% дохода ({input_dict["Доход клиента"]:,} руб.)')
    if credit_to_income > 10:
        reject_reasons.append(f'Сумма кредита ({input_dict["Сумма кредита"]:,} руб.) превышает доход в {credit_to_income:.0f} раз')

    # Микрокредиты
    micro_loan_safe = (input_dict['Сумма кредита'] < 5000) and (payment_to_income < 0.05)

    # Корректировки риска
    risk_adjustment = 0.0
    adjustment_reasons = []

    # Правило 1: Молодой клиент + низкий доход (+10%)
    if input_dict['Возраст клиента'] < 25 and input_dict['Доход клиента'] < 20000:
        risk_adjustment += 0.10
        adjustment_reasons.append('Молодой возраст (<25) + низкий доход (<20k) → повышенный риск')

    # Правило 2: Среднее образование (+8%)
    if input_dict.get('Образование клиента') == 'Среднее':
        risk_adjustment += 0.08
        adjustment_reasons.append('Среднее образование → повышенный риск')

    # Правило 3: Мобильные телефоны + Среднее образование (+15%)
    if input_dict.get('Тип товара') == 'Мобильные телефоны' and input_dict.get('Образование клиента') == 'Среднее':
        risk_adjustment += 0.15
        adjustment_reasons.append('Мобильные телефоны + среднее образование → ОЧЕНЬ высокий риск')

    # Правило 4: Ювелирные украшения + Молодой возраст (+12%)
    if input_dict.get('Тип товара') == 'Ювелирные украшения' and input_dict['Возраст клиента'] < 25:
        risk_adjustment += 0.12
        adjustment_reasons.append('Ювелирные украшения + молодой возраст (<25) → очень высокий риск')

    # Правило 5: Среднее образование + Другие регионы (доп. +4%)
    if input_dict.get('Образование клиента') == 'Среднее' and input_dict.get('Регион выдачи кредита') == 'Другие регионы':
        risk_adjustment += 0.04
        adjustment_reasons.append('Среднее образование + другие регионы → усиленный риск')

    # Правило 6: Доход 27-38k (+5%)
    if 27000 <= input_dict['Доход клиента'] <= 38000:
        risk_adjustment += 0.05
        adjustment_reasons.append('Доход 27-38k → повышенный риск')

    # Правило 7: Короткий срок кредита (-5%)
    if input_dict['Срок кредита'] <= 6:
        risk_adjustment -= 0.05
        adjustment_reasons.append('Короткий срок кредита (≤6 мес) → сниженный риск')

    # Правило 8: Высокий доход (-5%)
    if input_dict['Доход клиента'] > 60000:
        risk_adjustment -= 0.05
        adjustment_reasons.append('Высокий доход (>60k) → сниженный риск')

    # Правило 9: СПб регион (-4%)
    if input_dict.get('Регион выдачи кредита') == 'Санкт-Петербург или ЛО':
        risk_adjustment -= 0.04
        adjustment_reasons.append('Санкт-Петербург/ЛО → сниженный риск')


    return auto_reject, micro_loan_safe, risk_adjustment, reject_reasons, adjustment_reasons


def main():
    artifact, config, metrics = load_model()
    model = artifact['model']
    num_features = artifact['num_features']
    cat_features = artifact['cat_features']

    st.title('Модель оценки риска дефолта по кредиту')
    st.markdown('Введите данные клиента, чтобы получить прогноз вероятности дефолта.')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Параметры кредита')
        month = st.slider('Месяц выдачи кредита', 1, 12, 6)
        loan_amount = st.number_input('Сумма кредита (руб.)', min_value=1000, max_value=100000000, value=25000, step=1000)
        loan_term = st.slider('Срок кредита (мес.)', 3, 36, 12)

        st.subheader('Клиент')
        age = st.slider('Возраст клиента', 18, 90, 32)
        gender = st.selectbox('Пол клиента', config['categories']['Пол клиента'])
        education = st.selectbox('Образование клиента', config['categories']['Образование клиента'])
        children = st.selectbox('Наличие детей у клиента', config['categories']['Наличие детей у клиента'])

    with col2:
        st.subheader('Дополнительно')
        region = st.selectbox('Регион выдачи кредита', config['categories']['Регион выдачи кредита'])
        income = st.number_input('Доход клиента (руб.)', min_value=1000, max_value=100000000, value=28000, step=1000)
        marital = st.selectbox('Семейное положение', config['categories']['Семейное положение'])
        operator = st.selectbox('Оператор связи', config['categories']['Оператор связи'])
        product = st.selectbox('Тип товара', config['categories']['Тип товара'])
        bank_client = st.selectbox('Является ли клиентом банка', config['categories']['Является ли клиентом банка'])

    input_dict = {
        'Месяц выдачи кредита': month,
        'Сумма кредита': loan_amount,
        'Срок кредита': loan_term,
        'Возраст клиента': age,
        'Пол клиента': gender,
        'Образование клиента': education,
        'Тип товара': product,
        'Наличие детей у клиента': children,
        'Регион выдачи кредита': region,
        'Доход клиента': income,
        'Семейное положение': marital,
        'Оператор связи': operator,
        'Является ли клиентом банка': bank_client
    }

    if st.button('Получить прогноз', type='primary'):
        auto_reject, micro_loan_safe, risk_adj, reject_reasons, adj_reasons = check_business_rules(input_dict)

        if auto_reject:
            st.error('АВТОМАТИЧЕСКИЙ ОТКАЗ')
            for reason in reject_reasons:
                st.warning(reason)
            st.info('Данная заявка не соответствует базовым критериям андеррайтинга и не может быть одобрена.')
        else:
            df_input = load_and_engineer(input_dict)
            X = df_input[num_features + cat_features]

            proba_raw = model.predict_proba(X)[0, 1]

            proba_adjusted = max(0.0, min(1.0, proba_raw + risk_adj))
            
            if micro_loan_safe:
                proba = min(proba_adjusted, 0.05)
                micro_notice = True
            else:
                proba = proba_adjusted
                micro_notice = False

            threshold = metrics.get('Threshold', 0.36)
            decision = 'Высокий риск дефолта' if proba >= threshold else 'Низкий риск дефолта'

            st.divider()
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric('Вероятность дефолта', f'{proba:.1%}')
            col_res2.metric('Решение', decision)
            col_res3.metric('Порог одобрения', f'{threshold:.2f}')

            if risk_adj != 0 or micro_loan_safe:
                st.subheader('Корректировки риска')
                if micro_loan_safe:
                    st.success('Микрокредит (< 5000 руб., платёж < 5% дохода) — риск ограничен на 5%')
                for reason in adj_reasons:
                    sign = '+' if 'повышенный' in reason else '-'
                    st.info(f'{sign} {reason}')
                if proba_raw != proba_adjusted and not micro_loan_safe:
                    st.caption(f'Скор модели: {proba_raw:.1%} - Скор с корректировками: {proba_adjusted:.1%}')

            if micro_notice:
                st.success('Применено правило микрокредита: сумма < 5000 руб. и платёж < 5% дохода — риск скорректирован вниз.')

            st.progress(min(proba, 1.0), text=f'Уровень риска: {proba:.1%}')

            with st.expander('Почему такое решение?'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'влияние': shap_values[0]
                }).sort_values('влияние', key=abs, ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#E45756' if v > 0 else '#4C78A8' for v in importance['влияние']]
                ax.barh(importance['feature'][::-1], importance['влияние'][::-1], color=colors[::-1])
                ax.set_title('Топ-10 факторов, повлиявших на решение')
                ax.set_xlabel('Влияние на скор риска')
                st.pyplot(fig)

    st.divider()
    st.subheader('Качество модели (на тестовой выборке)')
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric('PR-AUC', f'{metrics["PR-AUC"]:.3f}')
    mcol2.metric('ROC-AUC', f'{metrics["ROC-AUC"]:.3f}')
    mcol3.metric('Precision', f'{metrics["Precision"]:.3f}')
    mcol4.metric('Recall', f'{metrics["Recall"]:.3f}')


if __name__ == '__main__':
    main()
