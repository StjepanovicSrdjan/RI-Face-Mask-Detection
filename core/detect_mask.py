import cv2
from keras.models import load_model
import numpy as np

model = load_model('drive/MyDrive/ColabNotebooks/face_mask_detection/model.h5')

img = cv2.imread('drive/MyDrive/ColabNotebooks/face_mask_detection/dataset/train/with_mask/with_mask_21.jpg')

resized=cv2.resize(img, (150, 150))
normalized=resized/255.0
image_array = np.array(normalized)
reshaped_array = image_array.reshape((-1, 150, 150, 3))
print(model.predict(reshaped_array))


