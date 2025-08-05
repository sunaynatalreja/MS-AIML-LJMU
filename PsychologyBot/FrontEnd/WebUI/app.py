from flask import Flask, request, jsonify, render_template,send_file, abort
import openai
import os
import sys
import json
import cv2
from base64 import b64decode
import numpy as np

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SignLanguageDetection')))
from inference import predict_from_frames
from flask_cors import CORS

app = Flask(__name__,static_folder='static')
CORS(app, resources={r"/static/*": {"origins": "*"}})

openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty input"}), 400

    try:
        response = openai.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:psychologist:BqH1zlqs",
            #model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Imagine you are Paul, a skilled psychologist specializing in logotherapy and cognitive-behavioral therapy. Your role is to engage users in meaningful, insightful, and natural conversations about their thoughts, feelings, behaviors, and life experiences. You display curiosity and unconditional positive regard, connecting past and present experiences while maintaining a natural conversational tone. Always ask clarifying or thought-provoking questions, gently provide advice or observations, and seek validation for your insights from the user. Keep the conversation dynamic by varying the topics you explore, such as free association, childhood, family dynamics, work, and hobbies. Never break character, avoid presenting information in lists, and ensure that the session continues seamlessly without ending until the user chooses to do so. Begin with a warm and open-ended question that encourages the user to share their thoughts or feelings. Conclude each response with a probing question that invites further discussion and deepens the conversation."},
                {"role": "user", "content": user_message}
            ],
        )
        reply = response.choices[0].message.content.replace("END", "").strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    frames = data['frames']
    decoded_frames = []
    for frame in frames:
        img_data = b64decode(frame.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decoded_frames.append(img)

    label = predict_from_frames(decoded_frames)  # returns label string
    return jsonify({'prediction': label})

@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    print("Serving video:", filename)
    video_path = os.path.join(app.static_folder, 'videos', filename)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4', conditional=True)
    else:
        abort(404)

if __name__ == '__main__':
    import mimetypes
    mimetypes.add_type('video/mp4', '.mp4')
    app.run(debug=True, port=5000)

