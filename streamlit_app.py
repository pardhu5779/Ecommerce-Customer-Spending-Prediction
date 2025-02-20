def main():
    st.title("Ecommerce Customer Spending Prediction")
    
    # Load data
    data = load_data()
    
    # Perform EDA
    perform_eda(data)
    
    # Build and evaluate the model
    model = build_model(data)
    
    # Prediction interface
    st.write("### Predict Yearly Amount Spent")
    
    avg_session_length = st.number_input("Average Session Length (minutes)", min_value=0.0)
    time_on_app = st.number_input("Time on App (minutes)", min_value=0.0)
    time_on_website = st.number_input("Time on Website (minutes)", min_value=0.0)
    length_of_membership = st.number_input("Length of Membership (years)", min_value=0.0)
    
    if st.button("Predict"):
        input_data = [[avg_session_length, time_on_app, time_on_website, length_of_membership]]
        prediction = model.predict(input_data)
        st.write(f"Predicted Yearly Amount Spent: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
