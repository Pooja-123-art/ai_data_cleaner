"""
AI-Powered Data Cleaning & Accurate Imputation Platform
Complete Code with Audit Tracking Logs & Excel File Output Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.impute import KNNImputer

# Page setup configuration
st.set_page_config(
    page_title="AI Data Cleaner & Imputer",
    page_icon="🧼",
    layout="wide"
)

# Title layout
st.markdown("# 🧼 AI-Driven Data Cleaner & Imputation Platform")
st.markdown("##### Upload any damaged CSV or Excel file, and our AI will accurately fill empty cells with an audit log tracking history report.")
st.markdown("---")

# File upload panel layout supporting both CSV and Excel structures
uploaded_file = st.file_uploader("📂 Upload your dataset file here", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # 1. Read input dataset based on file extension configuration
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
        st.success("✅ File loaded successfully into application cache grid configuration!")
        
        # Display original uploaded overview panel
        st.markdown("### 📋 Original Data Preview (With Missing Values)")
        st.dataframe(df.head(15), use_container_width=True)
        
        # Calculate overall empty matrix diagnostics variables
        total_missing = df.isnull().sum().sum()
        
        if total_missing == 0:
            st.info("🎉 Perfect! Your dataset structure does not contain any missing cells or empty rows context variables.")
        else:
            st.warning(f"⚠️ Total detected empty cells waiting for AI alignment actions: **{total_missing}**")
            
            # Action activation execution button wrapper element layout
            if st.button("🚀 Execute Smart AI Data Cleaning Imputation Process", type="primary"):
                with st.spinner("AI Engine structural analysis ongoing... Calculating optimal weights parameters..."):
                    
                    # Make explicit working copies
                    cleaned_df = df.copy()
                    
                    # 2. Track specific coordinate locations for audit changes log report table
                    audit_records = []
                    
                    # Separate numerical algorithms column indexes mappings logic arrays
                    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    # Audit collection process checking loop execution before data transformations mapping
                    for row_idx, row in cleaned_df.iterrows():
                        for col_name in cleaned_df.columns:
                            if pd.isnull(row[col_name]):
                                audit_records.append({
                                    "Row Position (Index)": row_idx + 1,
                                    "Column Name Identity": col_name,
                                    "Original State Status": "Empty / Null"
                                })
                    
                    # 3. Core AI Imputation Block Engine executions configurations
                    # Processing numerical parameters using Machine Learning KNN arrays logic patterns
                    if numeric_cols and cleaned_df[numeric_cols].isnull().sum().sum() > 0:
                        imputer = KNNImputer(n_neighbors=3)
                        cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])
                    
                    # Processing text categorical components values mapping updates 
                    for cat_col in categorical_cols:
                        if cleaned_df[cat_col].isnull().sum() > 0:
                            # Fill character structures string rows data with dataset mode top iteration index values
                            most_frequent_value = cleaned_df[cat_col].mode()[0] if not cleaned_df[cat_col].mode().empty else "N/A"
                            cleaned_df[cat_col] = cleaned_df[cat_col].fillna(most_frequent_value)
                    
                    # 4. Finalise Audit Log Reporting details cross-mapping calculations values parameters updates
                    audit_df = pd.DataFrame(audit_records)
                    updated_values_list = []
                    logic_applied_list = []
                    
                    for record in audit_records:
                        r_idx = record["Row Position (Index)"] - 1
                        c_name = record["Column Name Identity"]
                        val = cleaned_df.at[r_idx, c_name]
                        
                        # Set display formatting string contexts updates indexes variables properties
                        if isinstance(val, float):
                            updated_values_list.append(f"{val:.2f}")
                            logic_applied_list.append("KNN Machine Learning Cluster Pattern Prediction")
                        else:
                            updated_values_list.append(str(val))
                            logic_applied_list.append("High-Frequency Categorical Mode Replacement")
                            
                    audit_df["AI Replaced Value (Prediction)"] = updated_values_list
                    audit_df["Core Technical Optimization Logic"] = logic_applied_list
                    
                    # 5. Success screen compilation visualizations updates renders
                    st.balloons()
                    st.markdown("---")
                    st.markdown("### ✨ AI Imputed Clean Dataset View Visualization Panel")
                    st.dataframe(cleaned_df.head(15), use_container_width=True)
                    
                    # 6. Binary Excel buffer export compiler logic blocks streaming parameters
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        cleaned_df.to_excel(writer, index=False, sheet_name='AI Cleaned Data Frame')
                    
                    st.download_button(
                        label="📥 Download Cleaned Dataset File (Excel format .xlsx)",
                        data=excel_buffer.getvalue(),
                        file_name="AI_Imputed_Clean_Dataset.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # 7. Rendering the Main Intention: Audit Logs Changes Tables
                    st.markdown("---")
                    st.markdown("### 🔍 Granular Structural Audit Logs Tracking Report")
                    st.markdown("The platform keeps track of exact index positions where changes were deployed, including target prediction values replacement properties:")
                    st.dataframe(audit_df, use_container_width=True)
                    
    except Exception as e:
        st.error(f"❌ Structural compiling exception encountered inside application pipeline runtime context layers: {str(e)}")