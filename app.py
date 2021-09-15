#import pymysql
from flask import Flask, render_template, request, jsonify, redirect, send_file
from werkzeug.utils import secure_filename
import os
import time
import gan_process
import cv2

app = Flask(__name__)
# db = pymysql.connect(
#     host="127.0.0.1", port=3306, user="root", passwd="", db="web_test", charset="utf8"
# )
# cursor = db.cursor()

########
# 화면쏴주기(get)
# 데이터보여주기(get)
# 사진업로드(post)
# 사진다운로드(get)
########


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/down", methods=["GET"])
def down():
    return render_template("down.html")

# 사진업로드
@app.route("/fileUpload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        name_list = os.listdir('/home/gan_unmask/GAN_Service/static/before')
        l = len(name_list)
        f.filename = "before_"+f'{l+1}'.zfill(5)+".jpg"
        # 저장경로 + 파일명, secure_는 경로보안용
        f.save('/home/gan_unmask/GAN_Service/static/temp/01.jpg')
        before_temp = cv2.imread('/home/gan_unmask/GAN_Service/static/temp/01.jpg')
        cv2.imwrite('/home/gan_unmask/GAN_Service/static/before/'+secure_filename(f.filename), before_temp)

        gan_process.goGan()
        time.sleep(15)

    return render_template("down.html")


# 사진다운로드
@app.route("/download/")
def down_file():
    try:
        return send_file(
            "./static/temp/02.jpg",
            attachment_filename="unmasking.jpg",
            as_attachment=True
        )
    except Exception as e:
        return str(e) 

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0',port=80) #배포시에는 host='0.0.0.0' 꼭 필요
