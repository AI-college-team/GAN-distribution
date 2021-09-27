#import pymysql
from flask import Flask, render_template, request, jsonify, redirect, send_file
from werkzeug.utils import secure_filename
import os
import time
import gan_process2
import cv2
ganInstance = gan_process2.GAN_Service()
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

@app.route("/swap", methods=["GET"])
def swap():
    return render_template("swap.html")

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
        before_masked = before_temp.copy()
        cv2.imwrite('/home/gan_unmask/GAN_Service/static/before/'+secure_filename(f.filename), before_temp)

        ganInstance.goGan(before_masked)
        time.sleep(1)

    return render_template("down.html")

# 사진두장업로드
@app.route("/multifileUpload", methods=["GET", "POST"])
def upload_multifile():
    if request.method == "POST":
        #파일이 2개 넘어온다. 첫번째 파일
        f1 = request.files['file_m']
        name_list = os.listdir('/home/gan_unmask/GAN_Service/static/before')
        l = len(name_list)
        f1.filename = "before_"+f'{l+1}'.zfill(5)+".jpg"

        # 마스크드 사진 저장경로 + 파일명, 
        f1.save('/home/gan_unmask/GAN_Service/static/temp/01.jpg')
        before_temp = cv2.imread('/home/gan_unmask/GAN_Service/static/temp/01.jpg')
        before_masked = before_temp.copy()
        cv2.imwrite('/home/gan_unmask/GAN_Service/static/before_my/'+secure_filename(f1.filename), before_temp)


        #마스크 안쓴 사진, 번호는 마스크 쓴 사진과 동일하게 부여
        f2 = request.files["file_um"]
        f2.filename = "before_my_"+f'{l+1}'.zfill(5)+".jpg"
        # 언마스크드 사진 
        f2.save('/home/gan_unmask/GAN_Service/static/temp/03.jpg')
        before_temp2 = cv2.imread('/home/gan_unmask/GAN_Service/static/temp/03.jpg')
        before_unmasked = '/home/gan_unmask/GAN_Service/static/temp/03.jpg'
        cv2.imwrite('/home/gan_unmask/GAN_Service/static/before_my/'+secure_filename(f2.filename), before_temp2)

        ganInstance.goGanSwap(before_masked, before_unmasked)
        time.sleep(1)

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
