import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Import functions from existing modules
from sentiments import analyse_chats
from resolve import final_resolution
from insights import plot_sentiment_distribution, plot_customer_rating_distribution, plot_common_issues_clusters, plot_issue_type_distribution
import common_issue

# Load data
DATA_FILE = "chat_conversations_dataset.csv"

# Helper function to display matplotlib plots in Streamlit
def st_plot_matplotlib(plot_func, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_func(*args, **kwargs)
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Customer Chat Analysis Dashboard", layout="wide")
    st.title("📊 Customer Chat Analysis Dashboard")

    menu = ["Sentiment Analysis", "Issue Resolution", "Common Issues", "Insights"]
    choice = st.sidebar.selectbox("Navigate to", menu)

    data = pd.read_csv(DATA_FILE)

    if choice == "Sentiment Analysis":
        st.header("🔍 Sentiment Analysis")
        st.write(
            """
            Analyze sentiment of customer chat conversations using state-of-the-art transformer models.
            """
        )

        input_method = st.radio("Input Method", ("Select from dataset"))

        conversation_text = None

        if input_method == "Select from dataset":
            conv_options = data['full_conversation'].tolist()
            selected_conversation = st.selectbox("Select Conversation", conv_options, index=0)
            conversation_text = selected_conversation
        else:
            return None

        if st.button("Analyze Sentiment"):
            if conversation_text:
                with st.spinner('Analyzing sentiment...'):
                    result = analyse_chats(conversation_text)
                st.success("Analysis complete! See results below.")
                st.markdown("### Model Result")
                st.json(result["Model result"])
                st.markdown("### Customer's Actual Rating")
                st.metric(label="Rating", value=result["score_given by Model"])
                st.markdown("### Model Processing Time (seconds)")
                st.text(f"{result['Model time']:.4f} sec")
            else:
                st.warning("Please enter or select a conversation.")

    elif choice == "Issue Resolution":
        st.header("🛠️ Issue Resolution")
        st.write(
            """
            Get recommended resolution based on customer rating and issue description. 
            This module categorizes issues as common or uncommon and provides solutions or raises tickets accordingly.
            """
        )

        # Add conversation selection to auto-populate fields and provide resolution
        conv_options = data[['conversation_id', 'full_conversation']].apply(
            lambda row: f"ID {row['conversation_id']}: {row['full_conversation'][:75]}...", axis=1
        ).tolist()
        selected_conv = st.selectbox("Select Conversation to Auto-Fill", options=conv_options)

        # Extract conversation_id from selection
        selected_id = int(selected_conv.split(":")[0].split()[1])
        selected_row = data.loc[data['conversation_id'] == selected_id].iloc[0]

        customer_id = st.text_input("Customer ID", value=str(selected_row['conversation_id']))
        customer_rating = st.number_input("Customer Rating (1-5)", min_value=1, max_value=5, step=1,
                                          value=int(selected_row['customer_rating']))
        issue_description = st.text_area("Issue Description", height=150,
                                         value=selected_row['issue_description'])

        # Auto-generate resolution on selection change
        if st.button("Get Resolution"):
            if customer_id and customer_rating and issue_description:
                with st.spinner("Processing resolution..."):
                    resolution = final_resolution(customer_id, customer_rating, issue_description)
                st.success("Resolution generated!")
                st.info(resolution)
            else:
                st.warning("Please fill in all inputs.")

    elif choice == "Common Issues":
        st.header("📌 Common Issues Clustering")
        st.write(
            """
            This section displays clusters of common customer issues identified from the chat conversations dataset.
            """
        )

        st.markdown("### Frequency of Common Issue Clusters")
        st.json(common_issue.common_issues_frequency)

        for cluster_num in range(common_issue.num_clusters):
            with st.expander(f"Cluster {cluster_num} Sample Issues (Top 5)"):
                cluster_issues = common_issue.data[common_issue.data['cluster_label'] == cluster_num]['full_conversation'].tolist()
                for idx, issue in enumerate(cluster_issues[:5], 1):
                    st.write(f"{idx}. {issue}")

    elif choice == "Insights":
        st.header("📈 Insights & Visualizations")
        st.write(
            """
            Visual analytics of customer chat data, issue types, ratings, and sentiment distributions.
            """
        )

        # Arrange plots in two columns layout for better visual distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            fig = plot_sentiment_distribution()
            st.pyplot(fig)

            st.subheader("Common Issues Clusters Frequency")
            fig = plot_common_issues_clusters()
            st.pyplot(fig)

        with col2:
            st.subheader("Customer Ratings Distribution")
            fig = plot_customer_rating_distribution()
            st.pyplot(fig)

            st.subheader("Common vs Uncommon Issues Distribution")
            fig = plot_issue_type_distribution()
            st.pyplot(fig)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Chat Analysis Team")
    st.sidebar.markdown("© 2025")


if __name__ == "__main__":
    main()
