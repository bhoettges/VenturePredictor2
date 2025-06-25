import os
import joblib
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Load your trained models
XGB_MODEL_PATH = 'xgboost_multi_model.pkl'
RF_MODEL_PATH = 'random_forest_model.pkl'
model_xgb = joblib.load(XGB_MODEL_PATH)
model_rf = joblib.load(RF_MODEL_PATH)

# Feature order for each quarter
required_fields = [
    "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
    "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
]
quarters = ["Q1", "Q2", "Q3", "Q4"]

# Helper: parse input string to features (expects comma-separated numbers)
def parse_features(input_str):
    try:
        features = [float(x.strip()) for x in input_str.split(',')]
        return features
    except Exception:
        return None

def predict_arr_growth_xgb(features):
    X = np.array([features])
    prediction = model_xgb.predict(X)
    return prediction.tolist()

def predict_arr_growth_rf(features):
    X = np.array([features])
    prediction = model_rf.predict(X)
    return prediction.tolist()

def arr_tool_xgb(input_str):
    features = parse_features(input_str)
    if features is None or len(features) != 28:
        return "Please provide exactly 28 comma-separated features (7 per quarter for 4 quarters)."
    result = predict_arr_growth_xgb(features)
    return f"[XGBoost] Predicted ARR YoY growth for the next 4 quarters: {result}"

def arr_tool_rf(input_str):
    features = parse_features(input_str)
    if features is None or len(features) != 28:
        return "Please provide exactly 28 comma-separated features (7 per quarter for 4 quarters)."
    result = predict_arr_growth_rf(features)
    return f"[Random Forest] Predicted ARR YoY growth for the next 4 quarters: {result}"

def csv_tool(input_str, model_choice='xgboost'):
    # input_str: file path or CSV string
    try:
        if os.path.exists(input_str):
            df = pd.read_csv(input_str)
        else:
            from io import StringIO
            df = pd.read_csv(StringIO(input_str))
    except Exception as e:
        return f"Could not read CSV: {e}"
    # Try to extract the last 4 quarters and required fields
    try:
        if len(df) < 4:
            return "CSV must have at least 4 rows (quarters)."
        # Use the last 4 rows
        df_last4 = df.tail(4)
        features = []
        for _, row in df_last4.iterrows():
            for field in required_fields:
                features.append(float(row[field]))
        if len(features) != 28:
            return "CSV does not contain all required fields for 4 quarters."
        if model_choice == 'random_forest':
            result = predict_arr_growth_rf(features)
            return f"[Random Forest] Predicted ARR YoY growth for the next 4 quarters: {result}"
        else:
            result = predict_arr_growth_xgb(features)
            return f"[XGBoost] Predicted ARR YoY growth for the next 4 quarters: {result}"
    except Exception as e:
        return f"Error processing CSV: {e}"

# LangChain tools for direct feature input
arr_growth_tool_xgb = Tool(
    name="XGBoost ARR Growth Predictor",
    func=arr_tool_xgb,
    description="Predicts ARR YoY growth for the next 4 quarters using the XGBoost multi-output model. Input: 28 comma-separated features."
)
arr_growth_tool_rf = Tool(
    name="Random Forest ARR Growth Predictor",
    func=arr_tool_rf,
    description="Predicts ARR YoY growth for the next 4 quarters using the Random Forest multi-output model. Input: 28 comma-separated features."
)

# LangChain tool for CSV upload
csv_growth_tool_xgb = Tool(
    name="XGBoost CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, model_choice='xgboost'),
    description="Predicts ARR YoY growth for the next 4 quarters using XGBoost from a CSV file or CSV string."
)
csv_growth_tool_rf = Tool(
    name="Random Forest CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, model_choice='random_forest'),
    description="Predicts ARR YoY growth for the next 4 quarters using Random Forest from a CSV file or CSV string."
)

# Set up OpenAI LLM (requires OPENAI_API_KEY env variable)
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

agent = initialize_agent(
    tools=[arr_growth_tool_xgb, arr_growth_tool_rf, csv_growth_tool_xgb, csv_growth_tool_rf],
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

def interactive_data_entry(model_choice='xgboost'):
    print("\nLet's collect your data for the last 4 quarters. Please answer the following prompts:")
    features = []
    for i, q in enumerate(quarters):
        print(f"\n--- {q} ---")
        for field in required_fields:
            while True:
                val = input(f"{field}: ")
                try:
                    features.append(float(val))
                    break
                except ValueError:
                    print("Please enter a valid number.")
    if model_choice == 'random_forest':
        result = predict_arr_growth_rf(features)
        print(f"\n[Random Forest] Predicted ARR YoY growth for the next 4 quarters: {result}\n")
    else:
        result = predict_arr_growth_xgb(features)
        print(f"\n[XGBoost] Predicted ARR YoY growth for the next 4 quarters: {result}\n")

print("\nWelcome to the ARR Growth Chat Agent!\n")
print("You can choose to:")
print("1. Upload a CSV file or paste CSV content")
print("2. Enter data step by step (spoon-fed)")
print("3. Enter all features as a comma-separated list (raw)")
print("4. Exit\n")

while True:
    choice = input("Choose an option (1=CSV, 2=Step-by-step, 3=Raw, 4=Exit): ").strip()
    if choice == '1':
        model_choice = input("Which model? (xgboost/random_forest): ").strip().lower()
        csv_input = input("Enter CSV file path or paste CSV content: ")
        if model_choice == 'random_forest':
            print(csv_growth_tool_rf.run(csv_input))
        else:
            print(csv_growth_tool_xgb.run(csv_input))
    elif choice == '2':
        model_choice = input("Which model? (xgboost/random_forest): ").strip().lower()
        interactive_data_entry(model_choice)
    elif choice == '3':
        model_choice = input("Which model? (xgboost/random_forest): ").strip().lower()
        features_input = input("Enter 28 comma-separated features: ")
        if model_choice == 'random_forest':
            print(arr_growth_tool_rf.run(features_input))
        else:
            print(arr_growth_tool_xgb.run(features_input))
    elif choice == '4':
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.\n") 