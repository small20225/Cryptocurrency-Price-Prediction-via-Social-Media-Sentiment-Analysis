# Cryptocurrency Price Prediction via Social Media Sentiment Analysis

## Project Overview

This project develops a quantitative trading model that combines Twitter (X) social media sentiment analysis with cryptocurrency price prediction using deep learning techniques. The model analyzes social media sentiment to predict cryptocurrency price movements and generates trading signals.

## Key Features

- **Social Media Integration**: Processes Twitter (X) posts for sentiment analysis
- **Deep Learning Models**: Utilizes Deep Neural Networks (DNN) and LSTM for price prediction
- **Sentiment Analysis**: Implements NLTK VADER sentiment analysis and TF-IDF feature extraction

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Natural Language Processing**: NLTK (VADER Sentiment Analyzer)
- **Feature Extraction**: TF-IDF Vectorization
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Model Architecture**: Deep Neural Networks (DNN), LSTM

## Project Structure

```
Cryptocurrency-Price-Prediction-via-Social-Media-Sentiment-Analysis/
├── README.md                                    # Project documentation
├── requirements.txt                             # Required Python packages
├── train_model.py                              # Model training script
├── prediction.py                               # Prediction script
├── processed_data/                             # Processed datasets
├── models/                                     # Trained model files
│   ├── dnn_model_change_percentage.pth         # Trained DNN model
│   ├── scaler.pth                             # Data scaler
│   ├── std.pth                                # Standard deviation parameters
│   └── tfidf.pth                              # TF-IDF vectorizer
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/small20225/Cryptocurrency-Price-Prediction-via-Social-Media-Sentiment-Analysis.git
   cd Cryptocurrency-Price-Prediction-via-Social-Media-Sentiment-Analysis
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage

### Training the Model

Run the training script to train the deep learning model:

```bash
python train_model.py
```

This will:
- Load and preprocess the social media and price data
- Extract sentiment features using VADER and TF-IDF
- Train the DNN model
- Save the trained model and preprocessing objects

### Making Predictions

Use the prediction script to generate trading signals:

```bash
python prediction_new.py
```

This will:
- Load the pre-trained model and preprocessing objects
- Process new social media data
- Generate price movement predictions
- Output trading recommendations

## Model Architecture

- **Input Features**: 
  - Sentiment scores from social media posts
  - TF-IDF features from text content
  - Technical indicators from price data

- **Model Type**: Deep Neural Network (DNN)
- **Framework**: PyTorch
- **Output**: Cryptocurrency price change percentage

## Performance

- **Backtesting Period**: 4 years
- **Strategy**: Sentiment-driven trading based on social media analysis

## Data Sources

- **Social Media**: Twitter (X) posts related to cryptocurrencies
- **Price Data**: Historical cryptocurrency prices
- **Processing**: Sentiment analysis and feature engineering

## Key Components

1. **Sentiment Analysis**: NLTK VADER for real-time sentiment scoring
2. **Feature Engineering**: TF-IDF vectorization for text features
3. **Deep Learning**: PyTorch neural networks for price prediction
4. **Data Preprocessing**: StandardScaler for feature normalization

## Files Description

- `train_model.py`: Main training script with data preprocessing and model training
- `prediction.py`: Prediction script for generating trading signals
- `processed_data/`: Contains cleaned and processed datasets
- `models/`: Stores trained model weights and preprocessing objects

## Future Improvements

- Real-time data streaming integration
- Additional cryptocurrency pairs
- Enhanced feature engineering
- Model ensemble techniques
- Risk management integration

## Contact

**Chang Hsu**
- GitHub: [@small20225](https://github.com/small20225)

## License

This project is for educational and research purposes.

---

*This project demonstrates the application of machine learning and natural language processing techniques in quantitative finance and algorithmic trading.*