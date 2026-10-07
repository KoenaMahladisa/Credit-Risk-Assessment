import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title='AI Creditworthiness Assessment System',
    page_icon='💳',
    layout='wide',
)


# Load Artifacts
@st.cache_resource
def load_artifacts():
  model = joblib.load('credit_model.joblib')
  scaler = joblib.load('scaler.joblib')
  model_columns = joblib.load('model_columns.joblib')
  return model, scaler, model_columns


try:
  model, scaler, model_columns = load_artifacts()
except Exception as e:
  st.error(
      f'Artifacts not found. Please run `train_model.py` first. Error: {e}'
  )
  st.stop()

# UI Header
st.title('💳 AI Creditworthiness Assessment System')
st.markdown(
    'Evaluate loan applicant credit risk, prediction score, and recommended'
    ' actions using machine learning.'
)

# Input Form
with st.form('credit_form'):
  st.subheader("📋 Enter Applicant's Details")

  col1, col2, col3 = st.columns(3)

  with col1:
    cbal = st.selectbox(
        'Checking Account Balance',
        ['0 <= Rs. < 2000', 'no checking account', 'Rs. >= 2000', 'Rs. < 0'],
    )
    cdur = st.slider('Loan Duration (Months)', 4, 72, 24)
    chist = st.selectbox(
        'Credit History',
        [
            'all settled till now',
            'none taken/all settled',
            'dues not paid earlier',
            'critical account',
            'delay in paying off',
        ],
    )
    cpur = st.selectbox(
        'Loan Purpose',
        [
            'electronics',
            'Business',
            'car',
            'furniture',
            'repairs',
            'education',
            'retraining',
            'others',
        ],
    )
    camt = st.number_input('Loan Amount (Rs.)', min_value=1000, value=31690)
    sbal = st.selectbox(
        'Savings Account',
        ['Rs. < 1000', 'no savings account', '1000 <= Rs. < 5000', 'Rs. >= 5000'],
    )

  with col2:
    edur = st.selectbox(
        'Employment Duration',
        [
            '1 to 4 years',
            'more than 7 years',
            'less than 1 year',
            '4 to 7 years',
            'unemployed',
        ],
    )
    inrate = st.slider('Interest / Installment Rate', 1, 4, 4)
    msg = st.selectbox(
        'Personal Status & Gender',
        [
            'married or widowed male',
            'single male',
            'divorced or separated or married female',
            'divorced or separated male',
        ],
    )
    oparties = st.selectbox(
        'Other Parties / Guarantors', ['no one', 'yes, guarantor', 'co-applicant']
    )
    rdur = st.slider('Present Residence Duration', 1, 4, 2)
    prop = st.selectbox(
        'Property',
        [
            'real estate',
            'Other cars etc.',
            'life insurance/building society',
            'Unknown',
        ],
    )
    age = st.number_input('Age', min_value=18, max_value=100, value=26)

  with col3:
    inplans = st.selectbox('Other Installment Plans', ['none', 'bank', 'stores'])
    htype = st.selectbox('Housing Type', ['own', 'free', 'rent'])
    numcred = st.slider('Existing Credits at this Bank', 1, 10, 1)
    jobtype = st.selectbox(
        'Job Type',
        [
            'employee with official position',
            'employed either in management, self or in high qualification',
            'unemployed/unskilled - non-resident',
            'skilled employee / official',
        ],
    )
    nddepend = st.slider('Number of Dependents', 1, 5, 1)
    telephone = st.selectbox('Telephone Registered', ['yes', 'no'])
    foreign = st.foreign_input = st.selectbox('Foreign Worker', ['no', 'yes'])

  submitted = st.form_submit_button('Assess Creditworthiness')

# Processing and Prediction Logic
if submitted:
  input_data = pd.DataFrame(
      {
          'Cbal': [cbal],
          'Cdur': [cdur],
          'Chist': [chist],
          'Cpur': [cpur],
          'Camt': [camt],
          'Sbal': [sbal],
          'Edur': [edur],
          'Inrate': [inrate],
          'Msg': [msg],
          'Oparties': [oparties],
          'Rdur': [rdur],
          'Prop': [prop],
          'Age': [age],
          'Inplans': [inplans],
          'Htype': [htype],
          'Numcred': [numcred],
          'Jobtype': [jobtype],
          'Ndepend': [nddepend],
          'Telephone': [telephone],
          'Foreign': [foreign],
      }
  )

  # Preprocess user inputs identical to model pipeline
  input_encoded = pd.get_dummies(input_data, drop_first=True)
  for col in model_columns:
    if col not in input_encoded.columns:
      input_encoded[col] = 0
  input_encoded = input_encoded[model_columns]

  input_scaled = scaler.transform(input_encoded)

  prediction = model.predict(input_scaled)[0]
  proba = model.predict_proba(input_scaled)[0]
  confidence = np.max(proba) * 100

  credit_status = 'GOOD CREDIT' if prediction == 1 or prediction else 'BAD CREDIT'
  risk_level = (
      'LOW'
      if confidence > 75 and credit_status == 'GOOD CREDIT'
      else ('HIGH' if credit_status == 'BAD CREDIT' else 'MEDIUM')
  )

  # Display Results Dashboard
  st.markdown('---')
  st.subheader('-------------------------------------')
  st.subheader('        CREDIT ASSESSMENT')
  st.subheader('-------------------------------------')

  col_res1, col_res2, col_res3 = st.columns(3)
  with col_res1:
    st.metric(label='Prediction', value=credit_status)
  with col_res2:
    st.metric(label='Risk Level', value=risk_level)
  with col_res3:
    st.metric(label='Model Confidence', value=f'{confidence:.1f}%')

  st.markdown('### Recommended Action:')
  if credit_status == 'GOOD CREDIT':
    st.success('✓ Applicant may proceed to further assessment / loan approval.')
  else:
    st.error(
        '✗ Application flagged as high risk. Recommend manual underwriting or'
        ' additional review.'
    )

  st.markdown('### Key Factors Analyzed:')
  st.markdown(
      '- **Credit History & Status:** Payment track record and account balances'
      f'[cite: 1]\n- **Loan Characteristics:** Amount requested relative to'
      f' duration[cite: 1]\n- **Stability:** Employment duration and housing'
      f' type[cite: 1]'
  )