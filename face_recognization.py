from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import cv2
import os

# ファイルの場所 ------------------------------------------------------------------

#モデルの保存場所
MODEL_PATH="Models"
#顔のモデルの場所
FACE_MODEL = os.path.join(MODEL_PATH, "face-detection-retail-0004.xml")
FACE_WEIGHT= os.path.join(MODEL_PATH, "face-detection-retail-0004.bin")

# -------------------------------------------------------------------------------

def finding_faces(img_path, exec_net):   #絵文字の貼り付け    

    how_many_faces, data=0, []            #その写真に含まれる顔の数、座標、感情を保存するための変数
    #画像読み込みの一連の処理
    frame=cv2.imread(img_path)
    img_face = cv2.resize(frame, (300, 300))    # イメージサイズの変更 
    img_face = img_face.transpose((2, 0, 1))    # HWC から CHW 
    img_face = np.expand_dims(img_face, axis=0) # 次元合せ 

    # 推論実行(おまじない)
    out_face = exec_net.infer(inputs={'data': img_face})
    out_face = np.squeeze(out_face['detection_out'])

    for detection in out_face:              # 検出されたすべての顔領域に対して１つずつ処理 

        confidence = float(detection[2])    # confidence値の取得 

        # バウンディングボックス座標を入力画像のスケールに変換 
        x_1, y_1, x_2, y_2 = int(detection[3] * frame.shape[1]), int(detection[4] * frame.shape[0]), \
            int(detection[5] * frame.shape[1]), int(detection[6] * frame.shape[0])
        
        if confidence > 0.5:    # confidence値が0.5より大きい場合のみ顔だと判定
            
            #クリッピング処理
            x_1, y_1 = np.clip([x_1, y_1], 0, None)
            x_2, y_2 = np.clip([x_2], None, frame.shape[1])[0], np.clip([y_2], None, frame.shape[0])[0]
            
            data.append([x_1, y_1, x_2, y_2])   #顔の座標の代入
            how_many_faces+=1                   #その写真の顔の数を数える
    # 顔に緑色の四角形で囲む処理
        for a in range(how_many_faces): cv2.rectangle(frame,tuple(data[a][0:2]),tuple(data[a][2:4]),(0,255,0),3)

    return frame

def main():

    # モデルの読み込み（顔検出）
    plugin = IEPlugin(device="CPU")     # ターゲットデバイスの指定 
    net = IENetwork(model=FACE_MODEL, weights=FACE_WEIGHT)
    exec_net = plugin.load(network=net)

    files=os.listdir('Pre_Image')   #読み込む写真のファイル名取得
    files.remove('note.txt')

    for file in files:
        print(f'[INFO] File: {file}')

        img_path=os.path.join('Pre_Image', file)   #保存されている写真のパスを代入

        result=finding_faces(img_path, exec_net)   #顔の座標と表情指数判定

        cv2.imwrite(os.path.join('Post_Image', file), result)    #処理した写真を保存


if __name__ == '__main__':
    main()
