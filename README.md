AI-Driven Therapy Progress Tracker

 Overview
This repository contains the code for an AI-powered application designed to track and analyze progress in psychological therapy sessions. It leverages Natural Language Processing (NLP) to compare session notes, providing insights into patient progress through sentiment analysis, semantic similarity, and psychological assessment tools like GAD-7 and PHQ-9.

 Features
    - Session Comparison: Compare two therapy sessions to assess changes over time.
    
    - Sentiment Analysis: Analyze the emotional tone of session notes.
    
    - Semantic Similarity: Evaluate how similar session descriptions are.
    
    - Psychological Assessments: Automatic scoring of sessions based on anxiety (GAD-7) and depression (PHQ-9) scales.
    
    - Interactive User Interface: A Streamlit frontend for easy data input and result visualization.

 Tech Stack
- Backend: 
    - Python with FastAPI for the API server
      
    - Hugging Face Transformers for NLP tasks
      
    - SentenceTransformers for semantic comparison
      
- Frontend: 
  - Streamlit for the web interface

 Installation
 Prerequisites
- Python 3.8+
- pip

 Steps
 
 Clone the repository:
 
      sh
      
      git clone https://github.com/ojo007/therapy_progress_tracker.git
      
      cd therapy_progress_tracker

Install dependencies:
   
     pip install -r requirements.txt

Run the backend server:
   
      uvicorn app:app –reload

Run the frontend:
   
      streamlit run ui.py

Usage

    •	Upload session notes in JSON format using the Streamlit interface.
    
    •	Click 'Compare Sessions' to analyze and view the progress.

Project Structure

    •	app.py: Contains the FastAPI backend logic for processing session data.
    
    •	ui.py: Streamlit frontend for user interaction.
    
    •	requirements.txt: Lists all necessary Python packages.

