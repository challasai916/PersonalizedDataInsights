# Sample Datasets for Personalization Platform

This directory contains sample datasets that can be used to explore the personalization platform.

## Available Datasets

The platform can generate two types of sample datasets:

1. **E-commerce User Behavior**
   - User demographic data (age, gender)
   - Product information (category, price)
   - Order history (purchase date, quantity)
   - User ratings
   - RFM metrics (Recency, Frequency, Monetary value)

2. **Content Platform Engagement**
   - User demographic data (age, device)
   - Content information (type, category, duration)
   - Interaction data (views, likes, shares)
   - Engagement metrics (view duration, interaction frequency)

## How to Use

The sample datasets are generated programmatically when you select them from the "Data Upload & Processing" page.
You can load them by selecting the desired dataset type and clicking "Load Sample Dataset".

## Data Structure

### E-commerce Sample Dataset

| Column | Description |
|--------|-------------|
| user_id | Unique identifier for each user |
| product_id | Unique identifier for each product |
| order_date | Date when the order was placed |
| quantity | Number of items purchased |
| order_value | Value of the order |
| rating | User rating (1-5) for the product |
| age_group | Age group of the user |
| gender | Gender of the user |
| signup_date | Date when the user signed up |
| product_category | Category of the product |
| product_price | Price of the product |
| order_frequency | Number of orders made by the user |
| days_since_last_order | Days elapsed since the user's last order |
| total_spend | Total amount spent by the user |

### Content Platform Sample Dataset

| Column | Description |
|--------|-------------|
| user_id | Unique identifier for each user |
| content_id | Unique identifier for each content item |
| interaction_type | Type of interaction (View, Like, Share, Comment) |
| interaction_time | Timestamp of the interaction |
| view_duration | Duration (in seconds) for which content was viewed |
| score | User rating or score for the content |
| content_type | Type of content (Article, Video, Podcast, Image) |
| content_category | Category of the content |
| publication_date | Date when the content was published |
| length | Length of the content (for articles, in words) |
| duration | Duration of the content (for videos/podcasts, in seconds) |
| age_group | Age group of the user |
| primary_device | Primary device used by the user |
| interests | User interests (comma-separated) |
| signup_date | Date when the user signed up |
| total_interactions | Total number of interactions by the user |
| avg_view_duration | Average duration for which the user views content |
| interaction_frequency | Number of interactions per day since signup |
