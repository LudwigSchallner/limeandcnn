import cv2
import matplotlib.pyplot as plt
img = cv2.imread("./test.jpg")
img = cv2.resize(img, (299, 299))
img = img.reshape(3, 299,299)
plt.figure()
plt.imshow(img)
plt.show()
