import streamlit as st

introduction_1_page = st.Page("Introduction/1_What_Is_It.py", title = "Realty Compass")
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

page.run()