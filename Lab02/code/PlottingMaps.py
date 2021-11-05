import numpy as np
import math
import matplotlib.pyplot as plt

#Random matrices representing the six images
Image1 = np.random.randint(255, size=(50, 50))
Image2 = np.random.randint(255, size=(50, 50))
Image3 = np.random.randint(255, size=(50, 50))
Image4 = np.random.randint(255, size=(50, 50))
Image5 = np.random.randint(255, size=(50, 50))
Image6 = np.random.randint(255, size=(50, 50))

#Random matrices representing the ground truth for canopy heights (one per image)
GT1 = np.random.randint(50, size=(50, 50))
GT2 = np.random.randint(50, size=(50, 50))
GT3 = np.random.randint(50, size=(50, 50))
GT4 = np.random.randint(50, size=(50, 50))
GT5 = np.random.randint(50, size=(50, 50))
GT6 = np.random.randint(50, size=(50, 50))

#Random matrices representing the predicted or estimated canopy heights (one per image)
Canopy1 = np.random.randint(50, size=(50, 50))
Canopy2 = np.random.randint(50, size=(50, 50))
Canopy3 = np.random.randint(50, size=(50, 50))
Canopy4 = np.random.randint(50, size=(50, 50))
Canopy5 = np.random.randint(50, size=(50, 50))
Canopy6 = np.random.randint(50, size=(50, 50))

#Errors or Differences in Canopy Height (Prediction - Ground Truth)
Error1 = Canopy1 - GT1
Error2 = Canopy2 - GT2
Error3 = Canopy3 - GT3
Error4 = Canopy4 - GT4
Error5 = Canopy5 - GT5
Error6 = Canopy6 - GT6

plt.subplots(6, 4, figsize=(20, 30))
plt.suptitle('Comparing Estimated Canopy Heights with Ground Truth')
plt.subplots_adjust(top=0.95)

plt.subplot(6,4,1); plt.imshow(Image1); plt.ylabel('Image 1 (Training)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,2); plt.imshow(GT1, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,3); plt.imshow(Canopy1, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,4); plt.imshow(Error1, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,5); plt.imshow(Image2); plt.ylabel('Image 2 (Training)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,6); plt.imshow(GT2, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,7); plt.imshow(Canopy2, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,8); plt.imshow(Error2, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,9); plt.imshow(Image3); plt.ylabel('Image 3 (Training)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,10); plt.imshow(GT3, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,11); plt.imshow(Canopy3, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,12); plt.imshow(Error3, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,13); plt.imshow(Image4); plt.ylabel('Image 4 (Validation)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,14); plt.imshow(GT4, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,15); plt.imshow(Canopy4, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,16); plt.imshow(Error4, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,17); plt.imshow(Image5); plt.ylabel('Image 5 (Testing)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,18); plt.imshow(GT5, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,19); plt.imshow(Canopy5, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,20); plt.imshow(Error5, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,21); plt.imshow(Image6); plt.ylabel('Image 6 (Testing)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,22); plt.imshow(GT6, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,23); plt.imshow(Canopy6, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,24); plt.imshow(Error6, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.show()
