# # import streamlit as st
# # import joblib
# # model = joblib.load("spam_model.pkl")
# # vectorizer = joblib.load("tfidf_vectorizer.pkl")
# # st.title("📩 Spam SMS Detector")
# # message = st.text_area("Enter your message:")
# # if st.button("Check Message"):
# #         message_tfidf = vectorizer.transform([message])
# #         prediction = model.predict(message_tfidf)[0]

# #         if prediction == "spam":
# #          st.error("🚨 This message is SPAM!")
# #         else:
# #          st.success("✅ This message is HAM!")

# import streamlit as st
# import joblib

# # Load the trained model and TF-IDF vectorizer
# model = joblib.load("spam_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

# # Page configuration
# st.set_page_config(
#     page_title="Spam SMS Detector",
#     page_icon="📩",
#     layout="centered"
# )

# # Title
# st.title("📩 Spam SMS Detector")
# st.write("Enter an SMS message below to check whether it is spam or a normal message.")

# # Message input
# message = st.text_area(
#     "Enter your message:",
#     placeholder="Example: Congratulations! You have won a free prize..."
# )

# # Prediction button
# if st.button("🔍 Check Message"):
    
#     if message.strip() == "":
#         st.warning("⚠️ Please enter a message first.")
    
#     else:
#         # Convert message into TF-IDF features
#         message_tfidf = vectorizer.transform([message])

#         # Predict using Logistic Regression
#         prediction = model.predict(message_tfidf)[0]
#         probabilities = model.predict_proba(message_tfidf)[0]
#         spam_probability = probabilities[1]

#         # Display result
#         if prediction == "spam":
#            # st.error("🚨 This message is SPAM!")
#            st.error("🚨 This message is SPAM!")
#            st.write(f"Spam probability: {spam_probability:.2%}")
#         else:
#             #st.success("✅ This message is HAM (Normal).")
#             st.success("✅ This message is HAM (Normal).")
#             st.write(f"Spam probability: {spam_probability:.2%}") 



import streamlit as st
import joblib

# --------------------------------------------------
# Load trained model and TF-IDF vectorizer
# --------------------------------------------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="📩",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    '<div class="main-title">📩 Spam SMS Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-powered machine learning application for detecting spam SMS messages.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Project information
# --------------------------------------------------
with st.expander("ℹ️ About this project"):
    st.write(
        """
        This application uses Machine Learning to classify SMS messages as
        **Spam** or **Ham (Normal)**.

        **Machine Learning Model:** Logistic Regression  
        **Text Processing:** TF-IDF Vectorization  
        **Features:** Unigrams + Bigrams  
        **Model Accuracy:** 97.49%
        """
    )

# --------------------------------------------------
# Message input
# --------------------------------------------------
st.subheader("📝 Enter SMS Message")

message = st.text_area(
    "Message",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)

# --------------------------------------------------
# Prediction button
# --------------------------------------------------
if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        # Convert message into TF-IDF features
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_tfidf)[0]

        # Get probabilities
        probabilities = model.predict_proba(message_tfidf)[0]

        # Find probability according to model classes
        spam_index = list(model.classes_).index("spam")
        spam_probability = probabilities[spam_index]

        ham_index = list(model.classes_).index("ham")
        ham_probability = probabilities[ham_index]

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------
        st.subheader("📊 Prediction Result")

        if prediction == "spam":

            st.error("🚨 This message is SPAM!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2%}"
                )

            with col2:
                st.metric(
                    "Ham Probability",
                    f"{ham_probability:.2%}"
                )

            st.progress(spam_probability)

            st.warning(
                "⚠️ Be careful with this message. "
                "Avoid clicking unknown links or sharing personal information."
            )

        else:

            st.success("✅ This message is HAM (Normal).")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Ham Probability",
                    f"{ham_probability:.2%}"
                )

            with col2:
                st.metric(
                    "Spam Probability",
                    f"{spam_probability:.2%}"
                )

            st.progress(ham_probability)

            st.info(
                "ℹ️ This message appears to be a normal SMS."
            )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Spam SMS Detector | TF-IDF + Logistic Regression | "
    "Machine Learning Project"
)

