# Import Streamlit and other necessary libraries for UI and backend communication
import streamlit as st
import requests
import logging
import json

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set page configuration to make the layout wide
st.set_page_config(page_title="AI-Driven Therapy Progress Tracker", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
body {
    color: #333333;
    background-color: #f0f2f5;
}
.header {
    color: #2c3e50;
    font-size: 30px;
    text-align: center;
}
.subheader {
    color: #3498db;
    font-size: 24px;
}
.progress-box {
    background-color: #ecf0f1;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Display the title with custom styling
st.markdown('<p class="header">AI-Driven Therapy Progress Tracker</p>', unsafe_allow_html=True)

# Sidebar for file uploaders
with st.sidebar:
    st.markdown('## Session Data Upload')
    uploaded_file1 = st.file_uploader("Choose Session 1 (JSON or TXT)", type=["json", "txt"], key="session1")
    uploaded_file2 = st.file_uploader("Choose Session 2 (JSON or TXT)", type=["json", "txt"], key="session2")

# Main content area
if st.button('Compare Sessions', key="compare"):
    if uploaded_file1 is not None and uploaded_file2 is not None:
        with st.spinner('Processing sessions...'):
            try:
                # Prepare files for the API request
                files = {
                    'session1': (uploaded_file1.name, uploaded_file1.getvalue(), 'application/json'),
                    'session2': (uploaded_file2.name, uploaded_file2.getvalue(), 'application/json')
                }

                # Send POST request to the FastAPI backend
                response = requests.post("http://localhost:8000/compare_sessions/", files=files)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Parse the JSON response
                progress_data = response.json().get('progress', {})
                logger.debug(f"Received progress data: {progress_data}")

                # Display results
                st.markdown('<p class="subheader">AI-Driven Comparison Results</p>', unsafe_allow_html=True)

                # Setup columns for displaying information
                columns = st.columns(2)

                # Display assessment progress if available
                if 'Assessment' in progress_data:
                    with columns[0]:
                        st.markdown('<div class="progress-box">', unsafe_allow_html=True)
                        st.markdown('#### Assessment Progress')
                        assessment = progress_data['Assessment']
                        st.write(f"**Old Score:** {assessment['old_score']}")
                        st.write(f"**New Score:** {assessment['new_score']}")
                        st.write(f"**Progress Status:** {assessment['progress']}")
                        st.markdown('</div>', unsafe_allow_html=True)

                # Display individual analyses for other keys
                for key, details in progress_data.items():
                    if key != 'Assessment':
                        with columns[1 if 'Assessment' in progress_data else 0]:
                            st.markdown(f'<div class="progress-box"><h4>{key} Analysis</h4>', unsafe_allow_html=True)
                            if 'old_sentiment' in details:
                                # Sentiment analysis details
                                st.write(f"**Description:** {details['description']}")
                                st.write(f"**Old Sentiment:** {details['old_sentiment']:.2f}")
                                st.write(f"**New Sentiment:** {details['new_sentiment']:.2f}")
                                st.write(f"**Similarity Score:** {details['similarity']:.2f}")
                                st.write(f"**Progress Status:** {details['progress']}")
                            else:
                                # General progress details
                                st.write(f"**Old Value:** {details['old_value']}")
                                st.write(f"**New Value:** {details['new_value']}")
                                st.write(f"**Similarity Score:** {details['similarity']:.2f}")
                                st.write(f"**Progress Status:** {details['progress']}")
                            st.markdown('</div>', unsafe_allow_html=True)

            except requests.RequestException as e:
                st.error(f"Error connecting to the backend server: {str(e)}")
                logger.error(f"RequestException: {e}")
            except KeyError as e:
                st.error(f"Unexpected response format: Missing key {str(e)}")
                logger.error(f"KeyError: {e}")
            except json.JSONDecodeError as e:
                st.error(f"Response from server is not valid JSON: {str(e)}")
                logger.error(f"JSONDecodeError: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Unexpected Exception: {e}")
    else:
        st.error("Please upload both session files.")