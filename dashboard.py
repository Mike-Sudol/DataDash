import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest, f_classif


def main():
    st.set_page_config(
                       page_title="DataDash",
                       page_icon="📊",  # You can also use a URL to an image or a local file path
                       layout="wide")
    st.title("DataDash") 
    with open('./style.css') as f:
        css = f.read()
 
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv") 

    if uploaded_file is not None:
        # Check if 'df' is in session state, load new file if not or if different file
        if 'df' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name

        df = st.session_state.df

        # Display basic information
        st.header("Dataset Overview")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Shape of the dataset:")
            st.write(df.shape[0], "rows and", df.shape[1], "columns", "with ", df.isnull().sum().sum(), "missing values")
            st.write("Missing Values by Column:")
            st.write(df.isnull().sum()) 
            st.write("Features and their Data Types:")
            st.write(df.dtypes)

        with col2:
            st.write("Summary Statistics:")
            st.dataframe(df.describe())  # Using dataframe to better display tables
            st.write("Dataset Preview")
            st.dataframe(df.head())

        st.header("Data Visualization")
        col11, col12 = st.columns(2)

        with col11:
          st.subheader("Feature Distribution  Box Plot")
          box_column = st.selectbox("Select column for box plot", df.columns)
          if st.button("Generate Box Plot"):
              fig = px.box(df, y=box_column)
              st.plotly_chart(fig)

        with col12:
            st.subheader("Feature Distribution Histogram")
            hist_column = st.selectbox("Select column for histogram", df.columns)
            if st.button("Generate Histogram"):
                fig = px.histogram(df, x=hist_column)
                st.plotly_chart(fig)

         
        st.header("Actions") 
        col3, col4 = st.columns(2)

        with col3:  
            st.subheader("Select a column to replace missing values:")
            selected_column = st.selectbox("Choose column (select 'All' for all columns)", ['All'] + list(df.columns))
            fill_value = st.text_input("Enter fill value or leave empty to use mean/median/mode for numerical data")

            if st.button("Replace Missing Values"):
                if selected_column != 'All':
                    if fill_value:
                        df[selected_column].fillna(fill_value, inplace=True)
                    else: 
                        if pd.api.types.is_numeric_dtype(df[selected_column]):
                            fill_value = df[selected_column].mean()  # Can change to median() or mode()[0] as needed
                        else:
                            fill_value = df[selected_column].mode()[0]
                        df[selected_column].fillna(fill_value, inplace=True)
                else:
                    if fill_value:
                        df.fillna(fill_value, inplace=True)
                    else:
                        for column in df.columns:
                            if pd.api.types.is_numeric_dtype(df[column]):
                                df[column].fillna(df[column].mean(), inplace=True)  # Can change to median() or mode()[0] as needed
                            else:
                                df[column].fillna(df[column].mode()[0], inplace=True)

                # Update the DataFrame in session state after modification
                st.session_state.df = df
                st.success("Missing values replaced!")
                st.experimental_rerun()

        with col4:
          st.subheader("Apply One-Hot Encoding to categorical columns:")
          # Identify categorical columns
          categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
          selected_columns = st.multiselect("Select categorical columns", categorical_columns)
    
          if st.button("One-Hot Encode"):
              if selected_columns:
                  # Apply one-hot encoding
                  df_encoded = pd.get_dummies(df, columns=selected_columns, drop_first=True)
                  # Update DataFrame in session state
                  st.session_state.df = df_encoded
                  st.success(f"One-hot encoding applied to: {', '.join(selected_columns)}")
                  st.experimental_rerun()
              else:
                  st.error("No columns selected for one-hot encoding!")
     

        col5, col6v1  = st.columns(2)

        with col5:  
          st.header("Custom Python Code Execution")
          custom_code = st.text_area("Enter your custom Python code, data is stored as 'df' Pandas DataFrame") 

          if st.button("Run Custom Code"):
              exec(custom_code)
              st.success("Custom code executed")
              st.dataframe(st.session_state.df.head())
            
        with col6v1:
          st.subheader("Feature Selection")
          num_features = st.slider("Select number of features", min_value=1, max_value=len(df.columns)-1)
          target_column = st.selectbox("Select target column for feature selection", df.columns)

          if st.button("Select Features"):
              X = df.drop(columns=[target_column])
              y = df[target_column]
              selector = SelectKBest(score_func=f_classif, k=num_features)
              selector.fit(X, y)
              selected_columns = X.columns[selector.get_support()]
              df_selected = df[selected_columns]
              st.session_state.df = pd.concat([df_selected, y], axis=1)
              st.success(f"Selected top {num_features} features")
              st.dataframe(st.session_state.df.head())
        col6, col7, col8 = st.columns(3)

        with col6:
          st.subheader("Normalization/Scaling")
          scale_method = st.selectbox("Select scaling method", ["StandardScaler", "MinMaxScaler"])
          scale_columns = st.multiselect("Select columns to scale", df.select_dtypes(include=['float64', 'int64']).columns.tolist())
  
          if st.button("Apply Scaling"):
              if scale_columns:
                  if scale_method == "StandardScaler":
                      scaler = StandardScaler()
                  elif scale_method == "MinMaxScaler":
                      scaler = MinMaxScaler()
  
                  df[scale_columns] = scaler.fit_transform(df[scale_columns])
                  st.session_state.df = df
                  st.success(f"{scale_method} applied to: {', '.join(scale_columns)}")
                  st.experimental_rerun()
              else:
                  st.error("No columns selected for scaling!")
  
        with col7:
            st.subheader("Drop Columns")
            columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            if st.button("Drop Selected Columns"):
                if columns_to_drop:
                    df.drop(columns=columns_to_drop, inplace=True)
                    st.session_state.df = df
                    st.success(f"Columns dropped: {', '.join(columns_to_drop)}")
                    st.experimental_rerun()
                else:
                    st.error("No columns selected for dropping!")
  
        with col8:
            st.subheader("Remove Duplicate Rows")
            if st.button("Remove Duplicates"):
                df.drop_duplicates(inplace=True)
                st.session_state.df = df
                st.success("Duplicate rows removed!")
  
        col9, col10 = st.columns(2)
  
        with col10:
          st.subheader("Train-Test Split")
          test_size = st.slider("Select test size (percentage)", min_value=0.1, max_value=0.5, step=0.1, value=0.2)
          random_state = st.number_input("Random state", value=42, step=1)
  
          if st.button("Split Data"):
              train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
              st.write("Training set:")
              st.dataframe(train_df.head())
              st.write("Testing set:")
              st.dataframe(test_df.head())
              st.session_state.train_df = train_df
              st.session_state.test_df = test_df
              st.success("Data split into training and testing sets!")
  
              # Convert dataframes to CSV
              train_csv = train_df.to_csv(index=False).encode('utf-8')
              test_csv = test_df.to_csv(index=False).encode('utf-8')
  
              st.download_button(
                  label="Download Training Set CSV",
                  data=train_csv,
                  file_name="train_set.csv",
                  mime='text/csv',
              )
  
              st.download_button(
                  label="Download Testing Set CSV",
                  data=test_csv,
                  file_name="test_set.csv",
                  mime='text/csv',
              )
        with col9:
          st.subheader("Outlier Detection and Removal")
          outlier_method = st.selectbox("Select outlier detection method", ["Z-score", "IQR"])

          if st.button("Remove Outliers"):
              if outlier_method == "Z-score":
                  from scipy import stats
                  z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
                  abs_z_scores = np.abs(z_scores)
                  filtered_entries = (abs_z_scores < 3).all(axis=1)
                  df = df[filtered_entries]
              elif outlier_method == "IQR":
                  Q1 = df.quantile(0.25)
                  Q3 = df.quantile(0.75)
                  IQR = Q3 - Q1
                  df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

              st.session_state.df = df
              st.success(f"Outliers removed using {outlier_method} method")
              st.dataframe(df.head())

        # Save and Download Dataset Section
        st.header("Save and Download Dataset")
        save_filename = st.text_input("Enter filename for saving (with .csv extension)")
        if st.button("Save Dataset"):
          if save_filename:
              # Save the DataFrame to CSV 
              st.success(f"Dataset saved as {save_filename}")
  
              # Provide download link
              csv = df.to_csv(index=False).encode('utf-8')
              st.download_button(
                  label="Download CSV",
                  data=csv,
                  file_name=save_filename,
                  mime='text/csv',
              )
          else:
              st.error("Please provide a filename to save the dataset!")
 

        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
