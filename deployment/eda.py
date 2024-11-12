# import libraries
import pandas as pd
import numpy as np

# streamlit
import streamlit as st

# for visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def run():
    # title
    st.title("Exploratory Data Analysis - Hotel Review")

    # horizontal line
    st.write("---")

    # Banner
    st.image('eda.png')

    # section dataframe
    # load data
    data = pd.read_csv('tripadvisor_hotel_reviews.csv')
    df = pd.read_csv('grouped_data.csv')

    # Display dataset overview
    st.write("## Dataframe Overview")
    st.write(f"Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")
    st.dataframe(data.head())

    # subheader
    st.write("## Exploratory Data Analysis")

    # EDA 1
    # Title
    st.write("### Ratings Distribution")

    # Create instances to save the count of ratings and percentage
    rating_counts = data['Rating'].value_counts().sort_index()
    rating_percentages = rating_counts / rating_counts.sum() * 100 

    # Define colors for each rating
    colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'orange']

    # Create a figure with 2 side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Bar chart
    bars = rating_counts.plot(kind='bar', color=colors, ax=ax1)
    bars.bar_label(bars.containers[0], label_type='edge', fontsize=10)
    ax1.set_title('Count of Ratings')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y')
    ax1.set_xticklabels(bars.get_xticklabels(), rotation=0)  

    # Subplot 2: Pie chart
    ax2.pie(
        rating_percentages, 
        labels=rating_percentages.index, 
        autopct='%.2f%%', 
        startangle=90, 
        colors=colors 
    )
    ax2.set_title('Percentage of Ratings')

    # Display both plots using Streamlit
    plt.tight_layout()
    st.pyplot(fig)  # Change here to use Streamlit to display the figure

    st.write("1. Positive Ratings Dominate: Approximately 66.66% of reviewers gave ratings of 4 and 5, indicating a high level of satisfaction.")
    st.write("2. Low Dissatisfaction: Only 15.68% of reviewers provided ratings of 1 and 2, suggesting minimal negative experiences.")
    st.write("3. Opportunity for Improvement: 10.66% of reviewers rated 3, indicating room for enhancement in product or service quality.")

    # EDA 2
    # Title
    st.write("### Reviews WordCloud")
    
    # Combining reviews based on sentiment
    all_reviews = ' '.join(df['Review'].astype(str).tolist())
    negative_reviews = ' '.join(df[df['Sentiment'] == 'Negative']['Review'])
    neutral_reviews = ' '.join(df[df['Sentiment'] == 'Neutral']['Review'])
    positive_reviews = ' '.join(df[df['Sentiment'] == 'Positive']['Review'])    
    
    # Function to display word clouds
    def display_wordcloud(title, reviews, description):
        # Create a new figure for each word cloud
        fig, ax = plt.subplots(figsize=(10, 6))
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        # Display the word cloud first
        st.pyplot(fig)
        # Then display the description
        st.write(description)
    
    # Display all reviews word cloud
    display_wordcloud(
        title='All Reviews Word Cloud', 
        reviews=all_reviews,
        description="The word cloud shows frequent mentions of room, hotel, resort, and restaurant, indicating a focus on accommodations and dining. Positive terms like nice and good suggest overall satisfaction, while beach and people highlight leisure activities and social interactions, reflecting a generally positive guest experience centered on comfort and enjoyment."
    )
    
    # Display negative reviews word cloud
    display_wordcloud(
        title='Negative Reviews Word Cloud', 
        reviews=negative_reviews,
        description="The negative word cloud highlights issues related to staff, bad, and problem, indicating dissatisfaction with service and experiences. Frequent mentions of room and hotel suggest complaints about accommodations."
    )
    
    # Display neutral reviews word cloud
    display_wordcloud(
        title='Neutral Reviews Word Cloud', 
        reviews=neutral_reviews,
        description="The neutral word cloud reflects mixed sentiments, with words like nice, good, and beach, indicating average experiences related to service and amenities, particularly around the food and resort."
    )
    
    # Display positive reviews word cloud
    display_wordcloud(
        title='Positive Reviews Word Cloud', 
        reviews=positive_reviews,
        description="The positive word cloud emphasizes positive experiences with great, nice, and good, particularly in relation to breakfast, restaurant, and the overall ambiance, indicating strong satisfaction with the stay and facilities."
    )


if __name__ == "__main__":
    run()
