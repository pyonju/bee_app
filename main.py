import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 分類したいクラス名をclassesリストに格納
classes = ["スズメバチ", "アシナガバチ"]
# 学習に用いた画像のサイズを格納
image_size = 200

# アップロードされた画像を保存するフォルダ名
UPLOAD_FOLDER = "uploads"
# アップロードを許可する拡張子
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Flaskのインスタンスを作成
app = Flask(__name__)

# アップロードされたファイルの拡張子をチェックする関数を定義
# filenameの中に'.'という文字が存在するか、拡張子がALLOWED_EXTENSIONSの中に存在するか
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 学習済みモデルをロード
model = load_model('./bee_app.h5')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # POSTリクエストにファイルデータが含まれているか
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        # ファイルにファイル名があるか
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # ファイル名に危険な文字列がある場合に無効化(サニタイズ)
            filename = secure_filename(file.filename)
            # 与えられたパスにアップロードされた画像を保存する
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            # 保存先をfilepathに格納
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 受け取った画像を読み込み、np形式に変換
            # 画像のロードとリサイズを同時に行う(image.load_img)
            img = image.load_img(filepath, target_size=(image_size,image_size))
            # 引数に与えられた画像をNumpy配列に変換
            img = image.img_to_array(img)
            # predictに渡すため4次元配列に変換
            data = np.array([img])

            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            # 予測結果の最大値のインデックスを格納
            predicted = result.argmax()
            probability = round(result[predicted]*100, 1)
            pred_answer = "この写真は " + str(probability) + " %の確率で " + classes[predicted] + " です"

            # 引数にanswer=pred_answerを渡すことで、
            # index.htmlに書いたanswerにpred_answerを代入
            return render_template("index.html",answer=pred_answer)

    # POSTリクエストがない場合はindex.htmlのanswerに何も表示されない
    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)