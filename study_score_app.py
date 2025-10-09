import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

#---Sample data---

data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Score': [12,25,32,40,50,55,65,72,80,90]
}

df = pd.DataFrame(data)

#---Train model---

X = df[['Hours_Studied']]
y = df[['Score']]
model = LinearRegression()
model.fit(X, y)

#---Streamlit UI---

st.markdown("<h1 style='text-align:center;'>Study Hours vs Score predictor</h1>", unsafe_allow_html=True)
st.markdown("### Predict your exam score based on how many hours you studied")

#---Input---

hours = st.slider("Hours studied: ", 0.0, 10.0, 5.0, 0.5)
input_df = pd.DataFrame({'Hours_Studied': [hours]})
predicted_score = model.predict(input_df)[0][0]

#---Output---

#st.metric(label="Predicted score", value=f"{predicted_score: .2f}")
st.markdown(f"<h6 style='font-size: 26px; color: #FFC080'>Predicted Score: {predicted_score:.2f}</h6>", unsafe_allow_html=True)
st.markdown("### Study Trend")
st.line_chart(df.set_index('Hours_Studied'))

#---Footer---

st.markdown("---")
st.markdown("Made by Vaidehi Joshi ")
st.write("*Powered by caffeine *")


