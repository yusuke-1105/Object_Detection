'''
Copyright [yusuke-1105] [openvinotoolkit]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from openvino.inference_engine import IENetwork, IEPlugin
from math import exp
import numpy as np
import time
import cv2
import os

# ファイルの場所 ------------------------------------------------------------------

# モデルの保存場所
MODEL_PATH="Models"
# objectのモデルの場所
OBJECT_MODEL = os.path.join(MODEL_PATH, "yolov3.xml")
OBJECT_WEIGHT= os.path.join(MODEL_PATH, "yolov3.bin")
# ラベルの保存場所
COCO=os.path.join(MODEL_PATH, "coco.names")

# -------------------------------------------------------------------------------

class YoloParams_v3:
    # ------------------------------------------- layer parameters を抽出 ------------------------------------------
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.
'''
class TinyYoloParams_v3:
    # ------------------------------------------- layer parameters を抽出 ------------------------------------------
    # Magic numbers は yolo samples からコピー
	def __init__(self, param, side):
		self.num = 3 if 'num' not in param else len(param['mask'].split(',')) if 'mask' in param else int(param['num'])
		self.coords = 4 if 'coords' not in param else int(param['coords'])
		self.classes = 80 if 'classes' not in param else int(param['classes'])
		self.anchors = [float(a) for a in param['anchors'].split(',')]
		self.side = side
		if self.side == 13: self.anchor_offset = 2 * 3
		elif self.side == 26: self.anchor_offset = 2 * 0
		else: assert False, "Invalid output size. Only 13 and 26 " \
				"sizes are supported for output spatial dimensions"
'''
class TinyYolo_v3:
	@staticmethod
	def entry_index(side, coord, classes, location, entry):
		side_power_2 = side ** 2
		n = location // side_power_2
		loc = location % side_power_2
		return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

	@staticmethod
	def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
		xmin = int((x - w / 2) * w_scale)
		ymin = int((y - h / 2) * h_scale)
		xmax = int(xmin + w * w_scale)
		ymax = int(ymin + h * h_scale)
		return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

	@staticmethod
	def intersection_over_union(box_1, box_2):
		width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
		height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
		if width_of_overlap_area < 0 or height_of_overlap_area < 0: area_of_overlap = 0
		else: area_of_overlap = width_of_overlap_area * height_of_overlap_area
		box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
		box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
		area_of_union = box_1_area + box_2_area - area_of_overlap
		if area_of_union == 0: return 0
		return area_of_overlap / area_of_union
	@staticmethod
	def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, is_proportional):
		if is_proportional:
			scale = np.array([min(im_w/im_h, 1), min(im_h/im_w, 1)])
			offset = 0.5*(np.ones(2) - scale)
			x, y = (np.array([x, y]) - offset) / scale
			width, height = np.array([width, height]) / scale
		xmin = int((x - width / 2) * im_w)
		ymin = int((y - height / 2) * im_h)
		xmax = int(xmin + width * im_w)
		ymax = int(ymin + height * im_h)
		return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())

	@staticmethod
	def parse_yolo_region(blob, resized_image_shape, frameinal_im_shape, params, threshold):
		    # ------------------------------------------ output parameters を検証 ------------------------------------------
		_, _, out_blob_h, out_blob_w = blob.shape
		assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
										"be equal to width. Current height = {}, current width = {}" \
										"".format(out_blob_h, out_blob_w)

		# ------------------------------------------ layer parameters を抽出 -------------------------------------------
		orig_im_h, orig_im_w = frameinal_im_shape
		resized_image_h, resized_image_w = resized_image_shape
		objects = list()
		size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
		bbox_size = params.coords + 1 + params.classes
		# ------------------------------------------- YOLO Region output を解析 -------------------------------------------
		for row, col, n in np.ndindex(params.side, params.side, params.num):
			# Getting raw values for each detection bounding box
			bbox = blob[0, n*bbox_size:(n+1)*bbox_size, row, col]
			x, y, width, height, object_probability = bbox[:5]
			class_probabilities = bbox[5:]
			if object_probability < threshold: continue
			# Process raw value
			x = (col + x) / params.side
			y = (row + y) / params.side
			# Value for exp is very big number in some cases so following construction is using here
			try: width, height = exp(width), exp(height)
			except OverflowError: continue
			# Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
			width = width * params.anchors[2 * n] / size_normalizer[0]
			height = height * params.anchors[2 * n + 1] / size_normalizer[1]

			class_id = np.argmax(class_probabilities)
			confidence = class_probabilities[class_id]*object_probability
			if confidence < threshold: continue
			objects.append(TinyYolo_v3.scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
									im_h=orig_im_h, im_w=orig_im_w, is_proportional=0.15))
		return objects
	'''
	def parse_yolo_region(blob, resized_image_shape, frameinal_im_shape, params, threshold):
		# ------------------------------------------ output parameters を検証 ------------------------------------------
		_, _, out_blob_h, out_blob_w = blob.shape
		assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
										 "be equal to width. Current height = {}, current width = {}" \
										 "".format(out_blob_h, out_blob_w)

		# ------------------------------------------ layer parameters を抽出 -------------------------------------------
		frame_im_h, frame_im_w = frameinal_im_shape
		resized_image_h, resized_image_w = resized_image_shape
		objects = list()
		predictions = blob.flatten()
		side_square = params.side * params.side

		# ------------------------------------------- YOLO Region output を解析 -------------------------------------------
		for i in range(side_square):
			row = i // params.side
			col = i % params.side
			for n in range(params.num):
				obj_index = TinyYolo_v3.entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
				scale = predictions[obj_index]
				if scale < threshold: continue
				box_index = TinyYolo_v3.entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
				x = (col + predictions[box_index + 0 * side_square]) / params.side * resized_image_w
				y = (row + predictions[box_index + 1 * side_square]) / params.side * resized_image_h
				try: w_exp, h_exp = exp(predictions[box_index + 2 * side_square]), exp(predictions[box_index + 3 * side_square])
				except OverflowError: continue
				w = w_exp * params.anchors[params.anchor_offset + 2 * n]
				h = h_exp * params.anchors[params.anchor_offset + 2 * n + 1]
				for j in range(params.classes):
					class_index = TinyYolo_v3.entry_index(params.side, params.coords, params.classes, n * side_square + i,
						params.coords + 1 + j)
					confidence = scale * predictions[class_index]
					if confidence < threshold: continue
					objects.append(TinyYolo_v3.scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
						h_scale=frame_im_h / resized_image_h, w_scale=frame_im_w / resized_image_w))
		return objects
	'''

def object_detection(img_path, net, exec_net):

	# COCOデータセットのラベルのリスト。このリストに応じて、検出された「object」の名前を判断する
	LABELS = open(COCO).read().strip().split("\n")
	#「object」を検出して四角で囲む。その際の四角の色を決める
	COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))
	
	#おまじない----------------------------------
	inputBlob = next(iter(net.inputs))
	net.batch_size = 1
	n, c, h, w = net.inputs[inputBlob].shape
	prob_threshold, iou_threshold=0.5, 0.15

	frame = cv2.imread(img_path)
	img_object = cv2.resize(frame, (w, h))
	img_object = img_object.transpose((2, 0, 1))
	img_object = img_object.reshape((n, c, h, w))
	#--------------------------------------------

	# 推論実行(おまじない)
	output = exec_net.infer({inputBlob: img_object})
	objects = []

	# 推論の実行により、検出されたobjectの範囲を決める
	for (layerName, outBlob) in output.items():
		layerParams = YoloParams_v3(net.layers[layerName].params, outBlob.shape[2])
		objects += TinyYolo_v3.parse_yolo_region(outBlob, img_object.shape[2:], \
												frame.shape[:-1], layerParams, prob_threshold)

	# 検出したobjectsの信頼性が十分あるかを検証(本当に「object」かどうか)
	for i in range(len(objects)):
		if objects[i]["confidence"] == 0: continue
		for j in range(i + 1, len(objects)):
			if TinyYolo_v3.intersection_over_union(objects[i], objects[j]) > iou_threshold:
				objects[j]["confidence"] = 0
	objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

	# イメージ(全体)の縦の長さと横の長さをそれぞれ、endY、 endXに代入
	endY, endX = frame.shape[:-1]
	
	for obj in objects:
		# objectの座標がイメージの範囲を超えていないかを精査
		if obj["xmax"] > endX or obj["ymax"] > endY or obj["xmin"] < 0 or obj["ymin"] < 0: continue

		# そのobjectが何か、その可能性はどれくらいかを算出
		label=f"{LABELS[obj['class_id']]}: {obj['confidence'] * 100:.2f}%"

		# 上記の「label」の表記する座標を算出
		y = obj["ymin"] - 15 if obj["ymin"] - 15 > 15 else obj["ymin"] + 15

		# objectのあるところの周りに四角形を描いて、その上部には上記の「label」を表記
		cv2.rectangle(frame, (obj["xmin"], obj["ymin"]), (obj["xmax"], obj["ymax"]), COLORS[obj["class_id"]], 2)
		cv2.putText(frame, label, (obj["xmin"], y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[obj["class_id"]], 3)

	return frame

def main():
	
	# モデルの読み込み（顔検出） 
	plugin = IEPlugin(device="CPU")
	net = IENetwork(model=OBJECT_MODEL, weights=OBJECT_WEIGHT)
	exec_net = plugin.load(network=net, num_requests=1)

	#読み込む写真のファイル名取得
	files=os.listdir('Pre_Image')
	files.remove('note.txt')
	
	for file in files:
		print(f'[INFO] File: {file}')
		img_path=os.path.join('Pre_Image', file)		#保存されている写真のパスを代入
		result=object_detection(img_path, net, exec_net)	#object detectionを実行
		cv2.imwrite(os.path.join('Post_Image', file), result)	  #処理した写真を保存


if __name__ == '__main__':
    main()