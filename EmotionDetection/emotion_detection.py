import requests
import json

def emotion_detector(text_to_analyze):
    """
    Sends the input text to the IBM Watson Emotion Detection API and returns
    the scores for anger, disgust, fear, joy, and sadness, along with the dominant emotion.

    If the API returns a 400 status (e.g., blank input), returns all values as None.

    Parameters:
    - text_to_analyze (str): The text to analyze for emotions.

    Returns:
    - dict: A dictionary containing emotion scores and the dominant emotion,
            or all None values if input is invalid.
    """
    # API endpoint
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    # Payload
    payload = { "raw_document": { "text": text_to_analyze } }

    # Headers
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    # Send POST request
    response = requests.post(url, json=payload, headers=headers)

    # Handle 500 status code (e.g. blank input)
    if response.status_code == 500:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Convert the response text to dictionary
    result = json.loads(response.text)
    
    # Extract emotions from the response
    emotions = result['emotionPredictions'][0]['emotion']

    # Filter for required emotions
    required_emotions = {emotion: emotions[emotion] for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness']}

    # Find the dominant emotion
    dominant_emotion = max(required_emotions, key=required_emotions.get)
    
    # Add dominant emotion to the dictionary
    required_emotions['dominant_emotion'] = dominant_emotion

    return required_emotions
