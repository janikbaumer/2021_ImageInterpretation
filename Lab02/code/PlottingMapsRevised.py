import numpy as np
import math
import matplotlib.pyplot as plt

#Pre-processed image tiles
Tile0 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\RGBNIR_final_tile_0.npy")
Tile1 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\RGBNIR_final_tile_1.npy")
Tile2 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\RGBNIR_final_tile_2.npy")
Tile3 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\RGBNIR_final_tile_3.npy")
Tile4 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\RGBNIR_final_tile_4.npy")
Tile5 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\RGBNIR_final_tile_4.npy")

#Scale and normalize the values to be between 0 and 255
Tile0 = np.asarray(Tile0, dtype=np.uint8)
Tile1 = np.asarray(Tile1, dtype=np.uint8)
Tile2 = np.asarray(Tile2, dtype=np.uint8)
Tile3 = np.asarray(Tile3, dtype=np.uint8)
Tile4 = np.asarray(Tile4, dtype=np.uint8)
Tile5 = np.asarray(Tile5, dtype=np.uint8)

#Ground truth for canopy heights (one per image)
GT0 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\label_tile_0.npy")
GT1 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\label_tile_1.npy")
GT2 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\label_tile_2.npy")
GT3 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\label_tile_3.npy")
GT4 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\label_tile_4.npy")
GT5 = np.load(r"C:\Users\andre\Desktop\ETHZ\Engineering Geodesy and Photogrammetry\Image Interpretation\Labs\Lab 2\label_tile_5.npy")

#Note that ground truth values are good as is, do not need to be scaled

'''
#Matrices representing the predicted or estimated canopy heights (one per image)
Canopy0 = np.zeros((10980,10980))
Canopy1 = np.zeros((10980,10980))
Canopy2 = np.zeros((10980,10980))
Canopy3 = np.zeros((10980,10980))
Canopy4 = np.zeros((10980,10980))
Canopy5 = np.zeros((10980,10980))

#Errors or Differences in Canopy Height (Prediction - Ground Truth)
Error0 = Canopy0# - GT0
Error1 = Canopy1# - GT1
Error2 = Canopy2# - GT2
Error3 = Canopy3# - GT3
Error4 = Canopy4# - GT4
Error5 = Canopy5# - GT5

plt.subplots(6, 4, figsize=(20, 30))
plt.suptitle('Comparing Estimated Canopy Heights with Ground Truth')
plt.subplots_adjust(top=0.95)

plt.subplot(6,4,1); plt.imshow(Tile0); plt.ylabel('Image 1 (Training)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,2); plt.imshow(GT0, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,3); plt.imshow(Canopy0, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,4); plt.imshow(Error0, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,5); plt.imshow(Tile1); plt.ylabel('Image 2 (Training)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,6); plt.imshow(GT1, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,7); plt.imshow(Canopy1, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,8); plt.imshow(Error1, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,9); plt.imshow(Tile2); plt.ylabel('Image 3 (Training)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,10); plt.imshow(GT2, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,11); plt.imshow(Canopy2, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,12); plt.imshow(Error2, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,13); plt.imshow(Tile3); plt.ylabel('Image 4 (Validation)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,14); plt.imshow(GT3, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,15); plt.imshow(Canopy3, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,16); plt.imshow(Error3, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,17); plt.imshow(Tile4); plt.ylabel('Image 5 (Testing)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,18); plt.imshow(GT4, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,19); plt.imshow(Canopy4, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,20); plt.imshow(Error4, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.subplot(6,4,21); plt.imshow(Tile5); plt.ylabel('Image 6 (Testing)', fontsize = 14); plt.title('Sentinel-2 Imagery', fontsize = 14);
plt.subplot(6,4,22); plt.imshow(GT5, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(6,4,23); plt.imshow(Canopy5, cmap = 'inferno'); plt.title('Canopy Prediction'); cbar = plt.colorbar(); cbar.set_label('Estimated Canopy Height [m]', rotation=90);
plt.subplot(6,4,24); plt.imshow(Error5, cmap = 'bwr'); plt.title('Canopy Error'); cbar = plt.colorbar(); cbar.set_label('Canopy Height Difference [m]', rotation=90);

plt.show()

'''
'''
plt.subplots(2, 6, figsize=(20, 30))
plt.suptitle('Comparing Estimated Canopy Heights with Ground Truth')
plt.subplots_adjust(top=0.95)

plt.subplot(2,6,1); plt.imshow(Tile0); plt.title('Preprocessed Tile 0', fontsize = 14);
plt.subplot(2,6,2); plt.imshow(Tile1); plt.title('Preprocessed Tile 1', fontsize = 14);
plt.subplot(2,6,3); plt.imshow(Tile2); plt.title('Preprocessed Tile 2', fontsize = 14);
plt.subplot(2,6,4); plt.imshow(Tile3); plt.title('Preprocessed Tile 3', fontsize = 14);
plt.subplot(2,6,5); plt.imshow(Tile4); plt.title('Preprocessed Tile 4', fontsize = 14);
plt.subplot(2,6,6); plt.imshow(Tile5); plt.title('Preprocessed Tile 5', fontsize = 14);

plt.subplot(2,6,7); plt.imshow(GT0, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(2,6,8); plt.imshow(GT1, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(2,6,9); plt.imshow(GT2, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(2,6,10); plt.imshow(GT3, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(2,6,11); plt.imshow(GT4, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);
plt.subplot(2,6,12); plt.imshow(GT5, cmap = 'inferno'); plt.title('Canopy Ground Truth'); cbar = plt.colorbar(); cbar.set_label('True Canopy Height [m]', rotation=90);

plt.show()
'''

#Preprocessed Images from Sentinel-2
plt.subplots(1, 6, figsize=(40, 10))
plt.subplot(1,6,1); plt.imshow(Tile0); plt.title('Tile 0', fontsize = 14);
plt.subplot(1,6,2); plt.imshow(Tile1); plt.title('Tile 1', fontsize = 14);
plt.subplot(1,6,3); plt.imshow(Tile2); plt.title('Tile 2', fontsize = 14);
plt.subplot(1,6,4); plt.imshow(Tile3); plt.title('Tile 3', fontsize = 14);
plt.subplot(1,6,5); plt.imshow(Tile4); plt.title('Tile 4', fontsize = 14);
plt.subplot(1,6,6); plt.imshow(Tile5); plt.title('Tile 5', fontsize = 14);

plt.show()

#Canopy Height Ground Truth [m]
plt.subplots(1, 6, figsize=(40, 10))
plt.subplot(1,6,1); plt.imshow(GT0, cmap = 'inferno'); plt.title('Tile 0', fontsize = 14); cbar = plt.colorbar(); cbar.set_label('Height [m]', rotation=90);
plt.subplot(1,6,2); plt.imshow(GT1, cmap = 'inferno'); plt.title('Tile 1', fontsize = 14); cbar = plt.colorbar(); cbar.set_label('Height [m]', rotation=90);
plt.subplot(1,6,3); plt.imshow(GT2, cmap = 'inferno'); plt.title('Tile 2', fontsize = 14); cbar = plt.colorbar(); cbar.set_label('Height [m]', rotation=90);
plt.subplot(1,6,4); plt.imshow(GT3, cmap = 'inferno'); plt.title('Tile 3', fontsize = 14); cbar = plt.colorbar(); cbar.set_label('Height [m]', rotation=90);
plt.subplot(1,6,5); plt.imshow(GT4, cmap = 'inferno'); plt.title('Tile 4', fontsize = 14); cbar = plt.colorbar(); cbar.set_label('Height [m]', rotation=90);
plt.subplot(1,6,6); plt.imshow(GT5, cmap = 'inferno'); plt.title('Tile 5', fontsize = 14); cbar = plt.colorbar(); cbar.set_label('Height [m]', rotation=90);

plt.show()

