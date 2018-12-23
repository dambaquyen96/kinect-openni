import redis
import cv2
import time
import numpy as np

redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)

def get_image():
	img_data = redis_db.get('image')
	img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	return img

def get_toado():
	toado = np.array(redis_db.get('toado').split())
	toado = np.array(toado)
	toado = toado.reshape(int(len(toado)/5), 5)
	return toado

def draw_toado(img, toado):
	for row in toado:
		min_x = int(row[0])
		max_x = int(row[1])
		min_y = int(row[2])
		max_y = int(row[3])
		distance = float(row[4])
		cv2.rectangle(img,(min_x, min_y),(max_x, max_y),(0,255,0),2)
		cv2.putText(img,'Distance: {}'.format(distance), 
			    (min_x, min_y-10), 
			    cv2.FONT_HERSHEY_SIMPLEX, 
			    1, (0,255,0), 2)

while True:
	img = get_image()
	toado = get_toado()
	draw_toado(img, toado)
	cv2.imshow('test', img)
	k = cv2.waitKey(1)
	if k == ord('q'):
		break
