from time import sleep
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time 
import streamlit_antd_components as sac
from streamlit_condition_tree import condition_tree, config_from_dataframe


st.set_page_config(page_title="Litmus AutoML", page_icon=":brain:")
api_endpoint_login = ''
api_endpoint_table = 'http://192.168.0.232:1880/api/litmus/live/table'
api_endpoint_2 = 'http://192.168.0.232:1880/api/litmus/live/queriedData'


def next(): st.session_state.counter += 1
def prev(): st.session_state.counter -= 1
def get_table_name():
    return st.selectbox("Select Table", ["PCBFailureAnalysis", "enigparam"], key='table_selectbox')
def generate_breadcrumb(page):
    steps = ["&#x31;", "&#x32;", "&#x33;", "&#x34;", "&#x35;"]
    breadcrumb = '<div style="text-align:center;">'
    for i, step in enumerate(steps):
        if i + 1 == page:
            breadcrumb += f'<span style="color:green; font-size:36px;"><b>{step}</b></span>'
        else:
            breadcrumb += f'<span style="font-size:36px;"><b>{step}</b></span>'
        if i + 1 != len(steps):  # Add ' > ' if it's not the last step
            breadcrumb += '<span style="font-size:28px;"> &#x27FA; </span>'  # Increase the size of the arrow
    breadcrumb += '</div>'
    return breadcrumb
def validate_credentials(api_endpoint, username, password):
    if api_endpoint_login == 'http://192.168.0.232:1880/api/litmus/live/login':
        payload = {
            "username": username,
            "password": password
        }

        # Send a POST request to the API
        response = requests.post(api_endpoint, json=payload)
        
        if response.status_code == 200:
            alert = st.empty()
            alert.success("Connected successfully")
            st.session_state['connected'] = True
            time.sleep(1.75)
            alert.empty()

        else:
            alert = st.empty()
            alert.error("Failed to connect")
            st.session_state['connected'] = False
    else:
        alert = st.empty()
        alert.error("Invalid API endpoint")
        st.session_state['connected'] = False

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, 'Select', False)
    
    # Master checkbox to select/deselect all
    if st.checkbox('Select/Deselect All'):
        df_with_selections['Select'] = True
    else:
        df_with_selections['Select'] = False
    
    # Display data editor
    edited_df = st.data_editor(df_with_selections,
                               hide_index=True,
                               column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                               disabled=df.columns)
    
    # Retrieve rows where 'Select' is True
    selected_rows = edited_df[edited_df['Select']]
    
    # Store the selected rows in session state
    # st.session_state['selected_rows'] = selected_rows.index.tolist()
    st.session_state['selected_rows_data'] = selected_rows.drop('Select', axis=1).to_dict(orient='records')

    return selected_rows.drop('Select', axis=1)


def split_frame(input_df, rows):
    df = [input_df.loc[i:i+rows-1, :] for i in range(0, len(input_df), rows)]
    return df
    
def paginate_df(name: str, dataset, streamlit_object: str, disabled=None, num_rows=None):
    if dataset.empty:
        st.error("Provided DataFrame is empty. Cannot paginate an empty DataFrame.")
        return

    # Calculate batch size and total pages
    try:
        pagination = st.container()
        bottom_menu = st.columns((4, 1, 1))

        with bottom_menu[2]:
            batch_size = st.selectbox("Page size", options=[10, 25, 50, 100], key=f"{name}_batch_size")

        total_pages = max(1, (len(dataset) - 1) // batch_size + 1)

        with bottom_menu[1]:
            current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, key=f"{name}_current_page")

        with bottom_menu[0]:
            st.markdown(f"Page {current_page} of {total_pages}")

        start_index = (current_page - 1) * batch_size
        end_index = start_index + batch_size
        page_data = dataset.iloc[start_index:end_index]

        # Display the DataFrame in the appropriate container
        if not page_data.empty:
            if streamlit_object == "df":
                pagination.dataframe(page_data, height=300, use_container_width=True)  # Adjust height as needed
            elif streamlit_object == "editable df":
                # st.data_editor is not an existing Streamlit command, and should be replaced with the correct one
                # pagination.data_editor(page_data, num_rows=num_rows)  # Assuming data_editor is the correct command
                pass
        else:
            st.warning("No data to display on this page.")

    except Exception as e:
        st.error(f"An error occurred in paginate_df: {e}")

# Function to get and store selected rows in dataframe
def get_selected_indices(selections_df, session_key):
    selected_indices = selections_df.index[selections_df['Select']].tolist()
    st.session_state[session_key] = selected_indices
    return selections_df[selections_df['Select']].drop(columns=['Select'])

# Function to update checkbox state based on session state
def update_checkbox_state(index, key):
    return st.session_state[key].get(index, False)

@st.cache_data
def fetch_data(payload):
    response = requests.post(api_endpoint_2, json=payload)
    if response.status_code == 200:
        return response.json()

def next():
    # Check if on page 2 and if there are no items in selected_rows
    if not st.session_state['selected_rows_data']:
        st.warning("You must select at least one row before proceeding to the next page.")
    else:
        st.session_state['page'] += 1   
             
# Initialize the session state
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False

# Sidebar input fields
with st.sidebar:
    st.image('Litmus_Logo.png')
    st.image('autoML.png')
    st.title("Connect to Database")
    api_endpoint_login = st.text_input("API Endpoint", value="http://192.168.0.232:1880/api/litmus/live/login")
    username = st.text_input("Username", value="Tom")
    password = st.text_input("Password", value="abc123" ,type="password")
    connect_button = st.button("Connect")

    if connect_button:
        st.session_state['button_clicked'] = True
        validate_credentials(api_endpoint_login, username, password)         
        if st.session_state.get('connected', False):
            st.rerun()

     
# Page initialization     
if 'page' not in st.session_state:
    st.session_state['page'] = 1     
            
if st.session_state.get('connected', False) and st.session_state['button_clicked']:
    if st.session_state['page'] == 1:           
        if st.session_state['button_clicked']:
            st.markdown(generate_breadcrumb(st.session_state['page']), unsafe_allow_html=True)
            st.title("Step 1: Data Selection")
            
            table_name = get_table_name()
            payload = {}
            
            if 'previous_table_name' not in st.session_state or st.session_state['previous_table_name'] != table_name:
                st.session_state['items'] = []
                st.session_state['selected_items'] = []
                st.session_state['previous_table_name'] = table_name
                
            if st.button("Load Data", type="primary"):
                st.session_state['select_all'] = False
                if table_name == "PCBFailureAnalysis":
                    query_table = "getPCBFA"
                elif table_name == "enigparam":
                    query_table = "getPCBEnig"
                    
                payload["method"] = query_table
                response = requests.post(api_endpoint_table, json=payload) 
                    
                if response.status_code == 200:
                    json_data = response.json()
                    df = pd.DataFrame(json_data, index=range(len(json_data)))
                    print(df)
                    st.session_state['items'] = df.columns.tolist()
                    st.write('Selected table:', table_name)

            if 'items' in st.session_state and st.session_state['items']:
                if 'select_all' not in st.session_state:
                    st.session_state['select_all'] = False

                select_all = st.checkbox("Select All", value=st.session_state['select_all'], key='select_all_columns')

                if select_all != st.session_state['select_all']:
                    st.session_state['select_all'] = select_all
                    st.rerun()

                if st.session_state['select_all']:
                    st.session_state['selected_items'] = st.session_state['items']
                elif not st.session_state['select_all'] and 'selected_items' in st.session_state:
                    st.session_state['selected_items'] = []

                selected = st.multiselect('Select columns', st.session_state['items'], st.session_state['selected_items'], key='column_multiselect')
                st.session_state['selected_items'] = selected

                if st.session_state['selected_items']:
                    df_selected = pd.DataFrame(st.session_state['selected_items'], columns=['Selected Columns'])
                    st.dataframe(df_selected, use_container_width=True, hide_index=True)
                    # st.write('You selected: ', st.session_state['selected_items'])
                else:
                    st.write('No columns selected.')
                    
                if st.button("Next", key='next1', use_container_width=True):
                    if st.session_state['selected_items']:
                        st.session_state['page'] = 2  
                        st.rerun()
                    else:
                        st.warning("Please select at least one element before proceeding to the next page.")


if 'selected_rows_data' not in st.session_state:
    st.session_state['selected_rows_data'] = []

# Page 2
elif st.session_state['page'] == 2:
    st.markdown(generate_breadcrumb(st.session_state['page']), unsafe_allow_html=True)
    st.title("Step 2: Data Processing")
    
    # Query filtering section
    num_rows = st.number_input('No. of rows', value=3000, key='num_rows')
    # Assuming 'num_rows' is your input
    if 'num_rows' in st.session_state and st.session_state['num_rows'] != num_rows:
        st.session_state['query_num_rows'] = num_rows
        
    date_cols = st.columns(2)
    # Check if 'from_date' and 'to_date' exist in session state
    if 'from_date' not in st.session_state:
        st.session_state['from_date'] = None
    if 'to_date' not in st.session_state:
        st.session_state['to_date'] = None
        
    # Create temporary variables to hold the values of 'from_date' and 'to_date'
    from_date_temp = st.session_state['from_date']
    to_date_temp = st.session_state['to_date']

    with date_cols[0]:
        from_date_temp = st.date_input('From date', value=st.session_state['from_date'])
    with date_cols[1]:
        to_date_temp = st.date_input('To date', value=st.session_state['to_date'])
        
    # Update 'from_date' and 'to_date' in session state with the values of the temporary variables
    st.session_state['from_date'] = from_date_temp
    st.session_state['to_date'] = to_date_temp
        
    # Convert 'from_date' and 'to_date' to string format if they are not None
    from_date_str = st.session_state['from_date'].strftime('%Y-%m-%d') if st.session_state['from_date'] is not None else None
    to_date_str = st.session_state['to_date'].strftime('%Y-%m-%d') if st.session_state['to_date'] is not None else None
    
    # Initialize the conditions list if it doesn't exist
    if 'conditions' not in st.session_state:
        st.session_state['conditions'] = []
        
    # Use a numeric input to specify the number of conditions
    num_conditions = st.number_input('Conditions', min_value=0, value=len(st.session_state['conditions']))

    # If the number of conditions is greater than the length of the 'conditions' list in the session state, add new conditions
    while len(st.session_state['conditions']) < num_conditions:
        st.session_state['conditions'].append({
            'logical_operator': 'OR',
            'where_column': '',
            'where_operator': '',
            'where_value': ''
        })

    # If the number of conditions is less than the length of the 'conditions' list in the session state, remove conditions from the end
    while len(st.session_state['conditions']) > num_conditions:
        st.session_state['conditions'].pop()

    # Display the conditions
    for i, condition in enumerate(st.session_state['conditions']):
        where_columns = st.columns(5)
        if i != 0:  # No logical operator for the first condition
            with where_columns[0]:
                st.session_state['conditions'][i]['logical_operator'] = st.selectbox('Logical Operator', ['AND', 'OR'], use_container_width = True, key=f'logical_operator_{i}')
        with where_columns[1]:
            st.session_state['conditions'][i]['where_column'] = st.selectbox('Keyword', st.session_state['selected_items'], key=f'where_column_{i}')
        with where_columns[2]:
            st.session_state['conditions'][i]['where_operator'] = st.selectbox('Operator', ['=', '!=', '>', '<', '>=', '<='], key=f'where_operator_{i}')
        with where_columns[3]:
            st.session_state['conditions'][i]['where_value'] = st.text_input('Value', key=f'where_value_{i}')
            
        # Only execute if the number of conditions is at least 1
    if len(st.session_state['conditions']) >= 1:
        # Create a list to hold the conditions as strings
        where_conditions_str = []

        # Iterate over the conditions in st.session_state['conditions']
        for condition in st.session_state['conditions']:
            # Check if the 'where_value' is a number
            if condition['where_value'].isdigit():
                # If it's a number, don't add quotation marks
                condition_str = f"{condition['where_column']} {condition['where_operator']} {condition['where_value']}"
            else:
                # If it's not a number, add quotation marks
                condition_str = f"{condition['where_column']} {condition['where_operator']} '{condition['where_value']}'"
            # Add the condition string to the list
            where_conditions_str.append(condition_str)

        # Convert the list of condition strings to a single string with the selected logical operator as the separator
        where_clause = f" {condition['logical_operator']} ".join(where_conditions_str)
    else:
        where_clause = ""

    execute_query = st.button("Execute Query", key='execute_query')
        # Add 'from_date', 'to_date' and 'where_conditions' to the payload
    payload = {
        "table_name": st.session_state['previous_table_name'],
        "selected_items": st.session_state['selected_items'],
        "rows":  st.session_state['num_rows'],
        "from_date": from_date_str,
        "to_date": to_date_str,
        "where_clause": where_clause
    }
        
    try:
        response = requests.post(api_endpoint_2, json=payload)
        if response.status_code == 200:
            json_data = response.json()

            if json_data:
                df = pd.DataFrame(json_data)

                # Debug: Check if the DataFrame is empty after creation
                if not df.empty:
                    # paginate_df('My Dataframe', df, 'df', None, None)
                    selection = dataframe_with_selections(df)
                    st.write("Your selection")
                    st.write(selection)
                    st.session_state['selected_rows_data'] = selection.to_dict(orient='records')
                    
                    # Descriptive statistics
                    desc_states = selection.describe().transpose()
                    
                    # Additional statistics
                    desc_states['max'] = selection.max()
                    desc_states['median'] = selection.median()
                    desc_states['unique'] = selection.nunique()
                    desc_states['dtype'] = selection.dtypes
                    desc_states['null'] = selection.isnull().sum()
                    
                    # Rename index the 'features'
                    desc_states.index.name = 'Features'
                    
                    # Summary DataFrame
                    st.write("Descriptive Analytics Summary:")
                    st.dataframe(desc_states, use_container_width=True)
                    
                    
                    if 'show_heatmap' not in st.session_state:
                        st.session_state['show_heatmap'] = False
                        
                    # histogram
                    if 'histogram' not in st.session_state:
                        st.session_state['histogram'] = False
                    
                    # Create two columns
                    cols = st.columns(2)

                    # heatmap button
                    with cols[0]:
                        if st.button('Toggle Heatmap'):
                            st.session_state['show_heatmap'] = not st.session_state['show_heatmap']

                        if st.session_state['show_heatmap']:
                            # Calculate the correlation matrix
                            corr = selection.corr()

                            # Create a heatmap
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
                            st.pyplot(fig)
                            plt.close()

                    # histogram button
                    with cols[1]:
                        if 'histogram' not in st.session_state:
                            st.session_state['histogram'] = False

                        if st.button('Toggle Histogram'):
                            st.session_state['histogram'] = not st.session_state['histogram']

                        if st.session_state['histogram']:
                            # Create a histogram
                            fig, ax = plt.subplots(figsize=(10, 8))
                            pd.DataFrame(st.session_state['selected_rows_data']).hist(ax=ax)
                            st.pyplot(fig)
                            plt.close()
                    
                else:
                    st.error("Data loaded but DataFrame is empty.")
            else:
                st.error("Response was successful but no data was returned.")
        else:
            st.error(f"Failed to retrieve data: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while trying to connect to the API: {e}")

    cols = st.columns(2)
    with cols[1]:
        if st.button("Next", key='next2', use_container_width=True):
            st.session_state['page'] = 3
            st.session_state['from_date'] = None
            st.session_state['to_date'] = None
            st.rerun()
    with cols[0]:
        if st.button("Back", key='back2', use_container_width=True):
            st.session_state['page'] = 1
            st.session_state['from_date'] = None
            st.session_state['to_date'] = None
            st.rerun()
    
if 'target_feature' not in st.session_state:
    st.session_state['target_feature'] = None
if 'selected_training_features' not in st.session_state:
    st.session_state['selected_training_features'] = []
if 'p3_select_all' not in st.session_state:
    st.session_state['p3_select_all'] = False   
        
# Page 3
elif st.session_state['page'] == 3:
    st.session_state['p3_select_all'] = False
    st.markdown(generate_breadcrumb(st.session_state['page']), unsafe_allow_html=True)
    st.title("Step 3: Training/Target Feature Selection ")

    st.selectbox("Select Target Feature(Y)", [""] + st.session_state['selected_items'], key='target_feature')
    x_features = [feature for feature in st.session_state['selected_items'] if feature != st.session_state['target_feature']]

    training_features_selections = pd.DataFrame(x_features, columns=['Training Features(X)'])
    training_features_selections.insert(0, 'Select', False)

    # Master checkbox to select/deselect all
    st.session_state['p3_select_all'] = st.checkbox('Select/Deselect All', value=st.session_state['p3_select_all'])
    training_features_selections['Select'] = st.session_state['p3_select_all']

    # Display data editor    
    features_df = st.data_editor(training_features_selections, 
                                 use_container_width=True, 
                                 column_config={
                                     'Select': st.column_config.CheckboxColumn(
                                         required=True
                                     )
                                 },
                                 hide_index=True,
                                 key='features_df')

    selected_training_rows = features_df[features_df['Select']]

    st.session_state['selected_training_features'] = selected_training_rows.drop('Select', axis=1).to_dict(orient='records')
    
    cols = st.columns(2)
    with cols[1]:
        if st.button("Next", key='next3', use_container_width=True):
            st.session_state['page'] = 4  
            st.rerun()
    with cols[0]: 
        if st.button("Back", key='back3', use_container_width=True):
            st.session_state['page'] = 2  
            st.rerun()
            
# Page 4
elif st.session_state['page'] == 4:
    st.markdown(generate_breadcrumb(st.session_state['page']), unsafe_allow_html=True)
    st.title("Step 4: Model Selection and Training")
    st.session_state
    cols = st.columns(2)
    with cols[1]:
        if st.button("Next", key='next4', use_container_width=True):
            st.session_state['page'] = 5  
            st.rerun()
    with cols[0]: 
        if st.button("Back", key='back4', use_container_width=True):
            st.session_state['page'] = 3  
            st.rerun()
            
# Page 5
elif st.session_state['page'] == 5:
    st.markdown(generate_breadcrumb(st.session_state['page']), unsafe_allow_html=True)
    st.title("Step 5: Model Evaluation and Prediction")
    if st.button("Back", use_container_width=True):
        st.session_state['page'] = 4
        st.rerun()
     
     
     
