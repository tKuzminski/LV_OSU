import numpy as np
import matplotlib . pyplot as plt


img=plt.imread ('road.jpg')
img=img[ :,:,0].copy()
plt.figure()
plt.imshow( img , cmap ="gray")
plt.show ()

plt.figure()
plt.imshow( img , cmap ="gray",alpha=0.5)
plt.show ()

img_split=np.hsplit(img,4)
plt.figure()
plt.imshow( img_split[1] , cmap ="gray")
plt.show()


img_rotate=np.rot90(img)
plt.figure()
plt.imshow( img_rotate , cmap ="gray")
plt.show()


img_mirror=np.fliplr(img)
plt.figure ()
plt.imshow( img_mirror , cmap ="gray")
plt.show()
