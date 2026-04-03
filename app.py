import streamlit as st
import numpy as np
import pickle
import sqlite3

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# -------------------- DATABASE SETUP --------------------
conn = sqlite3.connect('churn_data.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS customers (
    credit_score INTEGER,
    geography TEXT,
    gender TEXT,
    age INTEGER,
    tenure INTEGER,
    balance REAL,
    products INTEGER,
    has_card INTEGER,
    active_member INTEGER,
    salary REAL,
    prediction INTEGER,
    probability REAL
)
''')

conn.commit()

# -------------------- SAVE FUNCTION --------------------
def save_data(data):
    c.execute('''
    INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()

# -------------------- LOAD MODEL --------------------
try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
except:
    st.error("❌ Model file not found. Make sure 'churn_model.pkl' is in the same folder.")
    st.stop()

# -------------------- TITLE --------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🏦 Bank Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center;'>Predict whether a customer will leave the bank</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- INFO --------------------
st.info("⚠️ Note: Model is trained on France, Germany, and Spain data. India is approximated.")

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("📊 Enter Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain", "India"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
active_member = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

# -------------------- ENCODING --------------------
geo_map = {
    "France": 0,
    "Germany": 1,
    "Spain": 2,
    "India": 0
}

gender_map = {"Male": 1, "Female": 0}

geo_encoded = geo_map[geography]
gender_encoded = gender_map[gender]

# -------------------- DISPLAY INPUT --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Profile")
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Age:** {age}")
    st.write(f"**Tenure:** {tenure} years")
    st.write(f"**Balance:** ${balance:,.2f}")
    st.write(f"**Salary:** ${salary:,.2f}")

with col2:
    st.subheader("⚙️ Account Details")
    st.write(f"**Geography:** {geography}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Products:** {num_products}")
    st.write(f"**Has Credit Card:** {has_card}")
    st.write(f"**Active Member:** {active_member}")

st.markdown("---")

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Churn"):

    input_data = np.array([[credit_score, geo_encoded, gender_encoded, age,
                            tenure, balance, num_products, has_card,
                            active_member, salary]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer will STAY\n\nProbability: {probability:.2f}")

    # Progress bar
    st.progress(float(probability))

    # Interpretation
    st.markdown("### 🧠 Interpretation")

    if probability > 0.7:
        st.write("🔴 High Risk: Immediate action required.")
    elif probability > 0.4:
        st.write("🟠 Medium Risk: Monitor customer.")
    else:
        st.write("🟢 Low Risk: Customer is stable.")

    # Business Insight
    st.markdown("### 💡 Business Insight")
    st.write("""
    - Customers with low activity are more likely to churn  
    - High balance customers may switch banks  
    - Active members are less likely to leave  
    """)

    # -------------------- SAVE DATA --------------------
    save_data((
        credit_score, geography, gender, age, tenure,
        balance, num_products, has_card, active_member,
        salary, int(prediction), float(probability)
    ))

    st.success("✅ Data saved successfully!")

# -------------------- SHOW STORED DATA --------------------
if st.checkbox("📂 Show Stored Data"):
    data = c.execute("SELECT * FROM customers").fetchall()
    st.write(data)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>💻 Built with Streamlit | ML Project</center>",
    unsafe_allow_html=True
)