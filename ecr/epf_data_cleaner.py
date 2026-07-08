
import pandas as pd
import numpy as np
import camelot

def clean_epf_data(df):
    """
    Clean EPF (Employee Provident Fund) data extracted from PDF using camelot.

    This function extracts the employee data table that appears after the 
    'ABRY benefit remarks' line and creates a clean DataFrame with proper column names.

    Args:
        df (pd.DataFrame): Raw DataFrame from camelot table extraction

    Returns:
        pd.DataFrame: Cleaned employee data with proper headers and data types
    """

    try:
        # Find the row with "ABRY benefit remarks"
        abry_row = None
        for i, row in df.iterrows():
            for col in df.columns:
                if pd.notna(row[col]) and 'ABRY benefit remarks' in str(row[col]):
                    abry_row = i
                    break
            if abry_row is not None:
                break

        if abry_row is None:
            raise ValueError("Could not find 'ABRY benefit remarks' in the data")

        # Find where employee data starts (look for first numeric serial number after ABRY)
        data_start_row = None
        for i in range(abry_row + 1, len(df)):
            first_col_val = df.iloc[i, 0]
            if pd.notna(first_col_val):
                try:
                    serial_num = int(float(first_col_val))
                    if serial_num == 1:  # Start of employee data
                        data_start_row = i
                        break
                except (ValueError, TypeError):
                    continue

        if data_start_row is None:
            raise ValueError("Could not find the start of employee data")

        # Find end of employee data (before reason codes)
        data_end_row = len(df)
        for i in range(data_start_row, len(df)):
            first_col_val = df.iloc[i, 0]
            if pd.notna(first_col_val) and ('Reason Code' in str(first_col_val) or 
                                          str(first_col_val).startswith('GK') or 
                                          str(first_col_val).startswith('EC')):
                data_end_row = i
                break

        # Extract employee data
        employee_data = df.iloc[data_start_row:data_end_row].copy()

        # Filter valid employee rows (those with numeric serial numbers)
        valid_employee_rows = []
        for idx, row in employee_data.iterrows():
            first_col_val = row.iloc[0]
            if pd.notna(first_col_val):
                try:
                    serial_num = int(float(first_col_val))
                    if serial_num > 0:   # Reasonable range for employee count
                        valid_employee_rows.append(idx)
                except (ValueError, TypeError):
                    continue

        if not valid_employee_rows:
            raise ValueError("No valid employee data found")

        # Keep only valid rows
        employee_data = employee_data.loc[valid_employee_rows]

        # Set proper column names based on EPF ECR format
        expected_columns = [
            'Sl_No', 'UAN', 'Name_as_per_ECR', 'Name_as_per_UAN_Repository',
            'Gross_Wages', 'EPF_Wages', 'EPS_Wages', 'EDLI_Wages',
            'EE_Contribution', 'EPS_Contribution', 'ER_Contribution', 'NCP_Days',
            'Refunds', 'Pension_Share', 'ER_PF_Share', 'EE_Share', 'Posting_Location'
        ]

        # Adjust column names based on actual number of columns
        num_cols = employee_data.shape[1]
        if num_cols >= len(expected_columns):
            new_columns = expected_columns + [f'Extra_Col_{i}' for i in range(len(expected_columns), num_cols)]
        else:
            new_columns = expected_columns[:num_cols]

        employee_data.columns = new_columns
        employee_data = employee_data.reset_index(drop=True)

        # Clean data types

        # 1. Serial Number
        if 'Sl_No' in employee_data.columns:
            employee_data['Sl_No'] = pd.to_numeric(employee_data['Sl_No'], errors='coerce').astype('Int64')

        # 2. UAN - format properly (remove scientific notation)
        if 'UAN' in employee_data.columns:
            employee_data['UAN'] = employee_data['UAN'].apply(
                lambda x: f"{int(float(x))}" if pd.notna(x) and str(x) != 'nan' else str(x)
            )

        # 3. Wage and contribution columns - clean and convert to numeric
        numeric_columns = ['Gross_Wages', 'EPF_Wages', 'EPS_Wages', 'EDLI_Wages', 
                          'EE_Contribution', 'EPS_Contribution', 'ER_Contribution', 'NCP_Days']

        for col in numeric_columns:
            if col in employee_data.columns:
                # Clean the column: remove commas, replace dashes with 0
                employee_data[col] = (employee_data[col].astype(str)
                                    .str.replace(',', '')
                                    .str.replace('-', '0'))
                # Convert to numeric
                employee_data[col] = pd.to_numeric(employee_data[col], errors='coerce').fillna(0).astype(int)

        # 4. Handle benefit columns (might contain dashes meaning no benefit)
        benefit_columns = ['Refunds', 'Pension_Share', 'ER_PF_Share', 'EE_Share']
        for col in benefit_columns:
            if col in employee_data.columns:
                employee_data[col] = (employee_data[col].astype(str)
                                    .str.replace('-', '0'))
                try:
                    employee_data[col] = pd.to_numeric(employee_data[col], errors='coerce').fillna(0).astype(int)
                except:
                    # Keep as string if numeric conversion fails
                    pass

        # 5. Text columns - clean
        text_columns = ['Name_as_per_ECR', 'Name_as_per_UAN_Repository', 'Posting_Location']
        for col in text_columns:
            if col in employee_data.columns:
                employee_data[col] = employee_data[col].astype(str).replace('nan', '')

        return employee_data

    except Exception as e:
        raise Exception(f"Error cleaning EPF data: {str(e)}")


def clean_epf_from_camelot(tables):
    """
    Clean EPF data directly from camelot table objects.

    Usage:
        tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice")
        cleaned_df = clean_epf_from_camelot(tables)

    Args:
        tables: camelot.core.TableList object from camelot.read_pdf()

    Returns:
        pd.DataFrame: Cleaned employee data
    """

    if len(tables) == 0:
        raise ValueError("No tables found in the PDF")

    # Try each table until we find one with ABRY benefit remarks
    target_table = None
    for i, table in enumerate(tables):
        temp_df = table.df

        # Check if this table contains the ABRY benefit remarks
        found_abry = False
        for idx, row in temp_df.iterrows():
            for col in temp_df.columns:
                if pd.notna(row[col]) and 'ABRY benefit remarks' in str(row[col]):
                    found_abry = True
                    break
            if found_abry:
                break

        if found_abry:
            target_table = temp_df
            break

    if target_table is None:
        # If no table contains ABRY benefit remarks, use the largest table
        target_table = max(tables, key=lambda x: x.df.shape[0]).df
        print("Warning: Could not find ABRY benefit remarks, using largest table")

    # Clean the identified table
    return clean_epf_data(target_table)


# Example usage function
def example_usage():
    """
    Example of how to use the cleaning functions with camelot
    """

    # Example usage with camelot
    # pdf_file = "your_epf_file.pdf"
    # tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice")
    # cleaned_df = clean_epf_from_camelot(tables)

    # Alternative: if you already have a DataFrame
    # cleaned_df = clean_epf_data(your_dataframe)

    # Save cleaned data
    # cleaned_df.to_csv('cleaned_epf_data.csv', index=False)

    print("Usage examples provided in comments above")

# Test our final function
print("Testing the final cleaning function:")
# result = clean_epf_data_final(df)
# print(f"Successfully extracted {len(result)} employee records")
# print(f"Columns: {list(result.columns)}")
