import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

image= misc.ascent()

image_transformed = np.copy(image)
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]

filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
weight = 1

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (image[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (image[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (image[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (image[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (image[x, y] * filter[1][1])
      output_pixel = output_pixel + (image[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (image[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (image[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (image[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if(output_pixel<0):
        output_pixel=0
      if(output_pixel>255):
        output_pixel=255
      image_transformed[x, y] = output_pixel

new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(image_transformed[x, y])
    pixels.append(image_transformed[x+1, y])
    pixels.append(image_transformed[x, y+1])
    pixels.append(image_transformed[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]
 
# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.show()
