import numpy as np
import matplotlib.pyplot as plt

FILEPATH_BASE = 'C:\scratch\ImgInt_Lab02_tile_'
FILEPATH_RGB_NIR_CORR = 'BLABLA'
TILES = [0, 1, 2, 3, 4, 5]

for tile in TILES:
    print(f'tile: {tile}')
    FILEPATH_RGB_NIR_CORR = f'{FILEPATH_BASE}{tile}.npy'
    FILEPATH_RGB_NIR_AVG = f'{FILEPATH_BASE}{tile}_averaged.npy'

    print('loading the file...')
    RGB_NIR_CORR = np.load(file=FILEPATH_RGB_NIR_CORR)
    print('loaded successfully')
    # n_orbits = RGB_NIR_CORR[0]

    # convert zero to NaN values
    RGB_NIR_CORR = np.where(RGB_NIR_CORR == 0, np.nan, RGB_NIR_CORR)

    # create a merged image from 20/30 orbiting images



    averaged_tile = np.nanmean(RGB_NIR_CORR, axis=0)
    print(f'saving tile {tile}')
    np.save(FILEPATH_RGB_NIR_AVG, averaged_tile)
    print(f'tile {tile} saved sucessfully')
print()