import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load data from a CSV or Excel file
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by handling missing values and scaling features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable (ARR YoY Growth)
    """
    # Create Quarter Num and Quarter Order from Financial Quarter
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['Quarter Order'] = (df['Year'] - 2000) * 4 + df['Quarter Num']
    df = df.sort_values(by=['id_company', 'Quarter Order']).reset_index(drop=True)
    
    # Filter for ARR 1M - 10M (assuming New ARR is in thousands)
    df['New ARR'] = df['New ARR'] * 1_000
    df = df[(df['New ARR'] >= 1_000_000) & (df['New ARR'] <= 10_000_000)].copy()
    
    # Define columns to use
    selected_columns = ['id_company', 'Quarter Order', 'ARR YoY Growth (in %)',
                       'Revenue YoY Growth (in %)', 'Gross Margin (in %)',
                       'EBITDA', 'Cash Burn (OCF & ICF)', 'LTM Rule of 40% (ARR)', 'Quarter Num']
    
    # Select only the columns we need
    df = df[selected_columns]
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Separate features and target (using ARR YoY Growth as target)
    X = df.drop('ARR YoY Growth (in %)', axis=1)
    y = df['ARR YoY Growth (in %)']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def create_timeseries_dataset(df):
    """
    Create time-series dataset for MultiOutput XGBoost (4 past quarters → predict next 4)
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        tuple: (X, Y) where X is the feature matrix and Y is the target matrix (4 quarters)
    """
    # Fill NaNs with median per company
    for col in df.columns:
        if col not in ['id_company', 'Quarter Order']:
            df[col] = df.groupby('id_company')[col].transform(lambda x: x.fillna(x.median()))
    
    # Create time-series dataset (4 past quarters → predict next 4)
    sequence_length = 4
    data = []
    
    for company in df['id_company'].unique():
        company_data = df[df['id_company'] == company].reset_index(drop=True)
        
        for i in range(len(company_data) - sequence_length - 3):  # Ensure 4 future quarters
            past_quarters = company_data.iloc[i:i+sequence_length]
            next_4 = company_data.iloc[i+sequence_length:i+sequence_length+4]
            
            row = list(past_quarters.drop(columns=['id_company', 'Quarter Order']).values.flatten())
            row.extend(next_4['ARR YoY Growth (in %)'].values)
            data.append(row)
    
    # Convert to DataFrame
    target_columns = ['Target_ARR_Q1', 'Target_ARR_Q2', 'Target_ARR_Q3', 'Target_ARR_Q4']
    input_features = df.columns[2:]  # Exclude id_company, Quarter Order
    columns = [f"{col}_Q{i+1}" for i in range(sequence_length) for col in input_features] + target_columns
    df_timeseries = pd.DataFrame(data, columns=columns)
    
    # Separate features and targets
    X = df_timeseries.drop(columns=target_columns).select_dtypes(include=['number'])
    Y = df_timeseries[target_columns]
    
    return X, Y

def get_processed_dataframe(df):
    """
    Get the processed dataframe with all necessary information for detailed analysis
    
    Args:
        df (pd.DataFrame): Original dataframe
        
    Returns:
        pd.DataFrame: Processed dataframe with company IDs and quarter information
    """
    # Create Quarter Num and Quarter Order from Financial Quarter
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['Quarter Order'] = (df['Year'] - 2000) * 4 + df['Quarter Num']
    df = df.sort_values(by=['id_company', 'Quarter Order']).reset_index(drop=True)
    
    # Filter for ARR 1M - 10M (assuming New ARR is in thousands)
    df['New ARR'] = df['New ARR'] * 1_000
    df = df[(df['New ARR'] >= 1_000_000) & (df['New ARR'] <= 10_000_000)].copy()
    
    # Define columns to use
    selected_columns = ['id_company', 'Quarter Order', 'ARR YoY Growth (in %)',
                       'Revenue YoY Growth (in %)', 'Gross Margin (in %)',
                       'EBITDA', 'Cash Burn (OCF & ICF)', 'LTM Rule of 40% (ARR)', 'Quarter Num']
    
    # Select only the columns we need
    df = df[selected_columns]
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target variable
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 