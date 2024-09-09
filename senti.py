import whisper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Step 1: Load Whisper model and transcribe audio
model = whisper.load_model("base")
result = model.transcribe(r"C:\Users\91720\OneDrive\Documents\text\harvard.wav")
transcribed_text = result['text']

# Step 2: Perform sentiment analysis on the transcribed text
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = analyzer.polarity_scores(transcribed_text)

# Extracting individual sentiment values
positive = sentiment_scores['pos']
neutral = sentiment_scores['neu']
negative = sentiment_scores['neg']

# Step 3: Plotting the sentiment scores
labels = ['Positive', 'Neutral', 'Negative']
values = [positive, neutral, negative]

plt.bar(labels, values, color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis of Transcribed Audio')
plt.ylabel('Sentiment Score')
plt.show()
