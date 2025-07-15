import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc  # type: ignore
import joblib # type: ignore
import plotly.graph_objects as go  
import sklearn

# st.write(f"scikit-learn version: {sklearn.__version__}")
# st.write(f"pandas version: {pd.__version__}")
# st.write(f"numpy version: {np.__version__}")# type: ignore

# Load Model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model_telco_logreg2.pkl")
        return model, None
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        st.error(error_msg)
        return None, error_msg

# Load the model
model, model_error = load_model()

# Streamlit App
st.title("Model Performance and Prediction UI")

# Check if model loaded successfully
if model is None:
    st.error("⚠️ Model could not be loaded. Please check the model file and dependencies.")
    st.info("Common issues:")
    st.write("- Model was trained with different library versions")
    st.write("- Missing dependencies in requirements.txt")
    st.write("- Incompatible Python versions")
    
    # Show library versions for debugging
    try:
        import sklearn
        import imblearn
        st.write(f"Current sklearn version: {sklearn.__version__}")
        st.write(f"Current imblearn version: {imblearn.__version__}")
    except ImportError as e:
        st.write(f"Import error: {e}")
    
    st.stop()

# Membuat tab
tab1, tab2 = st.tabs(["Model Performance Visualization", "Model Prediction UI"])

### TAB 1: MODEL PERFORMANCE VISUALIZATION
with tab1:
    st.subheader("Model Performance Visualization")

    # Input Data Test
    st.write("Upload test data (CSV) dengan fitur dan label untuk evaluasi model.")
    uploaded_file = st.file_uploader("Upload Test Data", type=["csv"], key="tab1_file")
    
    if uploaded_file is not None:
        try:
            # Membaca file
            test_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Test Data:")
            st.dataframe(test_data.head())

            # Split data menjadi fitur dan label
            if 'Churn' in test_data.columns:
                X_test = test_data.drop(columns=["Churn"])
                y_test = test_data["Churn"]

                # Pastikan label memiliki tipe yang sama
                y_true = y_test.replace({"No": 0, "Yes": 1})  # Konversi y_true ke 0 dan 1
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # 1. Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Blues")
                st.pyplot(fig)
                plt.close()

                # 2. Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                # 3. ROC Curve
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close()
                
                # 4. Comparison Plot
                st.subheader("Comparison Plot")
                fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)

                # Set the title for the whole figure
                fig.suptitle("Churn on Some Features", fontsize=16)

                # Create the countplots
                sns.countplot(x="OnlineSecurity", data=test_data, hue="Churn", ax=axes[0])
                axes[0].set_title("OnlineSecurity")
                axes[0].tick_params(axis='x', rotation=45)
                
                sns.countplot(x="OnlineBackup", data=test_data, hue="Churn", ax=axes[1])
                axes[1].set_title("OnlineBackup")
                axes[1].tick_params(axis='x', rotation=45)
                
                sns.countplot(x="DeviceProtection", data=test_data, hue="Churn", ax=axes[2])
                axes[2].set_title("DeviceProtection")
                axes[2].tick_params(axis='x', rotation=45)
                
                sns.countplot(x="TechSupport", data=test_data, hue="Churn", ax=axes[3])
                axes[3].set_title("TechSupport")
                axes[3].tick_params(axis='x', rotation=45)
                
                sns.countplot(x="InternetService", data=test_data, hue="Churn", ax=axes[4])
                axes[4].set_title("InternetService")
                axes[4].tick_params(axis='x', rotation=45)

                # Adjust layout
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)  # Adjust top space for suptitle

                # Display the plot in Streamlit
                st.pyplot(fig)
                plt.close()
                
            else:
                st.error("Data harus memiliki kolom 'Churn' sebagai target untuk evaluasi.")
                
        except Exception as e:
            st.error(f"Error processing test data: {str(e)}")

### TAB 2: MODEL PREDICTION UI
with tab2:
    st.subheader("Model Prediction UI")

    # Membuat sub-tab untuk single dan batch prediction
    sub_tab1, sub_tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    ## SINGLE PREDICTION
    with sub_tab1:
        st.subheader("Single Prediction")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Service Features")
            online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
            online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
            tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])

        with col2:
            st.subheader("Contract and Payment Features")
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=1000.0, value=50.0)
            duration_of_tenure = st.selectbox('Duration of Tenure', ['< 1 Year', '1 - 2 Years', '2 - 3 Years', '> 3 Years'])

        # Create prediction button
        if st.button('Predict Churn Probability', use_container_width=True):
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'InternetService': [internet_service],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'Contract': [contract],
                    'MonthlyCharges': [monthly_charges],
                    'DurationOfTenure': [duration_of_tenure]
                })

                # Make prediction
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)
                churn_prob = probability[0][1]

                # Create three columns for visualization
                result_col1, result_col2 = st.columns([2, 2])

                with result_col1:
                    st.subheader("Prediction Result")
                    if prediction[0] == 1:
                        st.error("⚠️ High Risk of Churn")
                    else:
                        st.success("✅ Low Risk of Churn")

                with result_col2:
                    st.subheader("Churn Probability")
                    # Create a gauge chart for probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob * 100,
                        title={'text': "Churn Risk"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': churn_prob * 100
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    ## BATCH PREDICTION
    with sub_tab2:
        st.subheader("Batch Prediction")
        st.write("Upload file CSV untuk batch prediction.")

        # File uploader untuk batch prediction
        uploaded_batch_file = st.file_uploader("Upload Batch File (CSV)", type=["csv"], key="tab2_file")

        if uploaded_batch_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_batch_file)
                st.write("Uploaded Data:")
                st.dataframe(batch_data.head())

                if st.button("Predict Batch"):
                    # Prediksi
                    predictions = model.predict(batch_data)
                    prediction_probs = model.predict_proba(batch_data)[:, 1]

                    # Tambahkan kolom hasil prediksi
                    batch_data["Prediction"] = predictions
                    batch_data["Prediction_Probability"] = prediction_probs

                    st.write("Prediction Results:")
                    st.dataframe(batch_data)

                    # Tombol untuk mengunduh hasil prediksi
                    st.download_button(
                        label="Download Predictions",
                        data=batch_data.to_csv(index=False).encode('utf-8'),
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing batch data: {str(e)}")
