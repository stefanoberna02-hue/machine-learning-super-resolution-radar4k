from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
test=np.linspace(0,255,256)
test=test.reshape(256,1)
print(test.shape)
print(test)
broad=np.ones((256,40))
print(broad.shape)
print(broad)
image_mtrx=broad*test
img=Image.fromarray((image_mtrx).astype(np.uint8))
# plt.imshow(img)
# plt.show()
broad=Image.fromarray((255*broad).astype(np.uint8))
plt.imshow(broad)
plt.show()


