"""Flask application for detecting emotions from user input using IBM Watson NLP."""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/")
def render_index_page():
    """Render the index HTML page with the input form."""
    return render_template('index.html')

@app.route("/emotionDetector")
def detect_emotion():
    """Handle emotion detection requests and return formatted response."""
    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')

    # If the input is missing or empty, return an error message
    if not text_to_analyze or not text_to_analyze.strip():
        return "Invalid text! Please try again!"

    # Call the emotion detector function
    result = emotion_detector(text_to_analyze)

    # If the dominant emotion is None (e.g. blank input or bad request), return error message
    if result['dominant_emotion'] is None:
        return "Invalid text! Please try again!"

    # Build and return formatted response
    formatted_message = (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is <b>{result['dominant_emotion']}</b>."
    )
    return formatted_message

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
