import cv2
import numpy as np

from tensorflow.keras import activations, models

from vis.utils import utils
from vis.visualization import visualize_saliency

last_layer_index = -1

def display_img(img):
  cv2.imshow("PRESS ANY KEY TO CLOSE", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
def display_combined_images(img1, img2, img3, img4):
  img = np.zeros((1024, 1024))  
  img[0:512,    0:512] =    img1.reshape((512,512))
  img[0:512,    512:1024] = img2.reshape((512,512))
  img[512:1024, 0:512] =    img3.reshape((512,512))
  img[512:1024, 512:1024] = img4.reshape((512,512))  
  display_img(img)

def get_model():  
  model_file_name = "model-data\\cnn-keras3.hd5"
  model = models.load_model(model_file_name)
  model.layers[last_layer_index].activation = activations.linear
  return utils.apply_modifications(model)

model = get_model()
img = utils.load_img("data\\scaled-cnn-data\\withQr\\page_101.tiff", target_size=(512, 512, 1))

# generate a heatmap that shows how pixel activations as normalized float values (in range [0,1])
grads = visualize_saliency(
  model = model, 
  layer_idx = last_layer_index, 
  filter_indices = None,
  seed_input = img, 
  backprop_modifier = "guided"
)

# convent the gradients to ints in range [0,255]
grad_ints = np.uint8(grads * 255)

# apply thresholding - https://en.wikipedia.org/wiki/Thresholding_(image_processing) 
ret, thresholded = cv2.threshold(grad_ints,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# apply dilation - https://en.wikipedia.org/wiki/Dilation_(morphology)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) #(13,13)
dilated = cv2.dilate(thresholded, kernel, iterations=1)

display_combined_images(img, grads, thresholded, dilated)

# get a bounding box to contain the salient image
connectivity = 4  
output = cv2.connectedComponentsWithStats(dilated, connectivity, cv2.CV_32S)
top = output[2][1, cv2.CC_STAT_TOP ] 
left = output[2][1, cv2.CC_STAT_LEFT ] 
width = output[2][1, cv2.CC_STAT_WIDTH ]
height = output[2][1, cv2.CC_STAT_HEIGHT ]
print("top, left, width, height", top, left, width, height)

img_qr = img[ top:(top+height), left:(left+width) ]
img_qr_padded = np.zeros((256, 256))
img_qr_padded[100:(100+height), 100:(100+width)] = img_qr.reshape((height, width))
display_img(img_qr_padded)












