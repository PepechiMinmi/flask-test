from flask import Flask, request, render_template
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # アップロードされたファイルを取得
    file1 = request.files['image1']
    file2 = request.files['image2']
    
    # 画像を読み込む
    img1 = cv2.imdecode(np.fromstring(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.fromstring(file2.read(), np.uint8), cv2.IMREAD_COLOR)

    # 画像をリサイズ
    width, height = 500, 500
    img1_resized = cv2.resize(img1, (width, height))
    img2_resized = cv2.resize(img2, (width, height))

    # 画像をフラットなベクトルに変換
    img1_flattened = img1_resized.flatten()
    img2_flattened = img2_resized.flatten()

    # コサイン類似度を計算
    similarity = cosine_similarity([img1_flattened], [img2_flattened])

    return f"Cosine Similarity: {similarity[0][0]}"

if __name__ == '__main__':
    app.run(debug=True)
