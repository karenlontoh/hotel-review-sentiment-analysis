# Hotel Reviews Sentiment Analysis 🏨💬🤖
Create a sentiment analysis model for hotel reviews using NLP techniques and TensorFlow, deployed to streamline feedback assessment and enhance customer experience.

## Introduction 🌍
Customer reviews play an essential role in the hospitality industry, impacting a hotel's reputation and influencing the decisions of potential guests. By analyzing customer sentiment through reviews, hotels can gain valuable insights into guest satisfaction, identify areas for improvement, and enhance their services. This project develops a deep learning model to classify hotel reviews based on ratings into sentiment categories (negative, neutral, positive), empowering hotel managers to take a data-driven approach to improve customer experiences.

## Dataset Overview 📊
The dataset for this project consists of customer reviews sourced from the TripAdvisor app, a platform for travelers to share experiences and find travel recommendations. The dataset includes:
- `Review`: Customer feedback in text form.
- `Rating`: A numerical score from 1 to 5 indicating satisfaction.

### Categories
- **Negative** (1-2)
- **Neutral** (3)
- **Positive** (4-5)

URL for the dataset: [TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)

## Methodology 🔍
1. **Data Loading and Inspection**: Load and inspect the dataset to ensure data quality.
2. **Exploratory Data Analysis (EDA)**: Analyze data trends and patterns.
3. **Feature Engineering**: Preprocess text, including tokenization, removal of stop words, and lemmatization.
4. **Model Training**:
   - Initial training using Artificial Neural Networks (ANN).
   - Improvements through hyperparameter tuning and transfer learning.
5. **Model Evaluation and Comparison**: Assess models and choose the best-performing one.
6. **Deployment**: Deploy the final model for practical usage.

## Project Objectives 🎯
Develop a high-accuracy model for sentiment classification to help hotel managers:
- Understand guest feedback more effectively.
- Identify strengths and weaknesses in service.
- Enhance overall customer experience and satisfaction.

## URL for Model Deployment 🚀
Explore the deployed model on Hugging Face: [Hotel Review Sentiment Analysis](https://huggingface.co/spaces/karenlontoh/hotel-review-sentiment-analysis)

## Model Analysis 🧮
### Strengths 💪
1. **High Accuracy**: The model's performance improved significantly, with accuracy increasing from 46.67% in the initial epoch to 92.69% by the final epoch, indicating effective learning.
2. **Consistent Loss Reduction**: Both training and validation loss decreased steadily over epochs, showing minimal overfitting and stable training.
3. **Stable Performance**: Good performance was maintained across both training and validation datasets, supporting model reliability.

### Weaknesses ⚠️
1. **Low Performance for Neutral Class**: The model struggles with the neutral class, reflected in a lower F1-score of 0.52, which highlights difficulties in differentiating neutral sentiments.
2. **Validation Accuracy Stagnation**: There was minimal improvement in validation accuracy after reaching a certain threshold, suggesting potential overfitting.
3. **Imbalanced Class Performance**: The model demonstrated better accuracy for negative and positive classes compared to the neutral class, which could point to data bias and uneven class representation.

### Improvements 🔧
To enhance the model's performance, consider the following adjustments:
1. **Increase Model Complexity**: Incorporate additional units in LSTM layers or add Conv1D layers to improve feature extraction.
2. **Tune Hyperparameters**: Experiment with various learning rates, batch sizes, and activation functions for optimized training.
3. **Adjust Regularization**: Implement different dropout rates and increase L2 regularization to prevent overfitting.
4. **Data Augmentation**: Enrich the dataset with more diverse samples to improve class balance.
5. **Cross-Validation**: Implement k-fold cross-validation to test the model's consistency across multiple data splits.
6. **Use Ensemble Methods**: Combine predictions from different models to create a robust final prediction.

## Conclusion 📈
The sentiment analysis model achieved high accuracy (92.69%) and consistent performance, proving effective for classifying hotel reviews. However, improvements are needed to address **overfitting** and enhance the classification of neutral sentiments. This model provides valuable insights for **hotel managers** to make informed decisions and refine **guest experience strategies**.

## Libraries Used 🛠️
- Pandas
- NumPy
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Seaborn

## Author 👩‍💻
Karen Lontoh  
LinkedIn: [Karmenia Ditabaya Lontoh](https://www.linkedin.com/in/karmenia-lontoh/)
