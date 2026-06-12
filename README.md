# Customer Chat Analysis Dashboard

## Project Description
The Customer Chat Analysis Dashboard is a Streamlit-based web application designed to provide comprehensive analysis and insights from customer chat conversations. It leverages state-of-the-art transformer models for sentiment analysis, clusters common issues using machine learning, offers automated issue resolution recommendations, and provides interactive visual analytics.

## Features Overview
- **Sentiment Analysis:** Analyze the sentiment of customer chat conversations using advanced transformer models, with options to select from existing dataset conversations or enter conversations manually.
- **Issue Resolution:** Get recommended resolutions based on customer ratings and issue descriptions. Automatically identifies if an issue is common or uncommon and provides solutions or raises support tickets accordingly.
- **Common Issues Clustering:** Visualize clusters of frequently reported customer issues identified through TF-IDF vectorization and KMeans clustering.
- **Insights Dashboard:** Interactive visualizations demonstrating sentiment distributions, customer rating distributions, issue cluster frequencies, and comparison of common vs uncommon issues.

## Tech Stack and Dependencies
The project utilizes the following key libraries and tools:
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Transformers (Huggingface)
- python-dotenv

All dependencies are listed in the `requirements.txt` file.

## Project Structure Overview
- `app.py`: Main Streamlit application file with the user interface and navigation.
- `sentiments.py`: Handles sentiment analysis using transformer models.
- `resolve.py`: Provides automated issue resolution based on customer ratings.
- `common_issue.py`: Performs clustering to identify common customer issues.
- `insights.py`: Generates data visualizations for insights dashboard.
- `chat_conversations_dataset.csv`: Dataset containing customer chat conversations and related metadata.
- `requirements.txt`: Python package dependencies list.

## Installation Instructions
1. Clone the repository to your local machine.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
Launch the Streamlit application with the following command:
```bash
streamlit run app.py
```
This will start a local server, and the dashboard can be accessed in your web browser at the URL provided in the terminal (usually http://localhost:8501).

## Usage Guide
- **Sentiment Analysis:** Select or enter a customer conversation and run the sentiment analysis model to see the sentiment result, customer rating, and processing time.
- **Issue Resolution:** Select a conversation to auto-fill customer ID, rating, and issue description fields. Click "Get Resolution" to receive recommendations or ticket information.
- **Common Issues:** Explore clusters of common issues derived from customer chat data, with sample conversations displayed per cluster.
- **Insights:** View various visualizations reflecting sentiment distribution, rating distributions, issue cluster frequencies, and the split between common and uncommon issues.

## Dataset Information
The application uses `chat_conversations_dataset.csv`, which contains anonymized customer chat conversations, customer ratings, issue descriptions, and other metadata to support analysis and insights.

## Developer / Team
Developed by Chat Analysis Team  
© 2025
 