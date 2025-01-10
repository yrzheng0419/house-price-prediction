import streamlit as st
import polars as pl
import pandas as pd


# st.header("House Price Prediction")

# st.markdown(
#     """
#     - Hello World!
#     """
# )

# st.logo('./images/banner.png', size = "large")

introduction_1_page = st.Page("Introduction/1_What_Is_It.py", title = "What Is It")
introduction_2_page = st.Page("Introduction/2_How_to_Use.py", title = "How to Use")

get_started_1_page = st.Page("Get_Started/1_Knowledge_Base.py", title = "Knowledge Base")
get_started_2_page = st.Page("Get_Started/2_House_Query.py", title = "House Query")

developing_1_page = st.Page("Learn_More/1_CHANGELOG.py", title = "CHANGELOG")
developing_2_page = st.Page("Learn_More/2_CONTRIBUTING.py", title = "CONTRIBUTING")
developing_3_page = st.Page("Learn_More/3_Contact_Us.py", title = "Contact Us")

page = st.navigation(
    {
        "Home": [introduction_1_page, introduction_2_page],
        "Get Started": [get_started_1_page, get_started_2_page],
        "Learn More": [developing_1_page, developing_2_page, developing_3_page]
    }
)

# dat = pl.read_csv('./dat/weight_diff.csv')

# client_green = st.selectbox(
#     'How many green land should near your house?',
#     pd.DataFrame(dat.select(pl.col('green').unique()))
# )

# 'The green land near the house: ', client_green

# @st.dialog("House Features")
# def hou_feature():
#     feature_select = st.selectbox(
#         'What features would you concern in house location?',
#         pd.Series(
#             dat.drop(['hex_id', 'x', 'y', ''])
#         )
#     )


# Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )



# if __name__ == '__main__':
#     main()

page.run()