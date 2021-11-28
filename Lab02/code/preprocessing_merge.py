import numpy as np
import matplotlib.pyplot as plt

def avg_and_save(RGBNIR, split, tile):
    averaged_tile = np.nanmean(RGBNIR, axis=0)
    print(f'saving split {split} of tile {tile}')
    fpath = f'C:\scratch\RGBNIR_red_tile_{tile}_split{split}'
    np.save(fpath, averaged_tile)
    print(f'split {split} of tile {tile} saved sucessfully')

FILEPATH_BASE = 'C:\scratch\ImgInt_Lab02_tile_'
TILES = [0, 1, 2, 3, 4, 5]
TILES = [5]
SPLIT = 5

for tile in TILES:
    print(f'tile: {tile}')
    FILEPATH_RGB_NIR_CORR = f'{FILEPATH_BASE}{tile}.npy'
    FILEPATH_RGB_NIR_AVG = f'{FILEPATH_BASE}{tile}_averaged.npy'

    print('loading the file...')
    RGB_NIR_CORR = np.load(file=FILEPATH_RGB_NIR_CORR)
    print('loaded successfully')
    n_orbits = RGB_NIR_CORR[0]
    #
    # convert zero to NaN values
    RGB_NIR_CORR = np.where(RGB_NIR_CORR == 0, np.nan, RGB_NIR_CORR)

    # create splits

    if tile in [0, 4, 5]:  # 30 orbits
        RGB_NIR_CORR_red_0 = RGB_NIR_CORR[0:5]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_0, split=0, tile=tile)

        RGB_NIR_CORR_red_1 = RGB_NIR_CORR[5:10]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_1, split=1, tile=tile)

        RGB_NIR_CORR_red_2 = RGB_NIR_CORR[10:15]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_2, split=2, tile=tile)

        RGB_NIR_CORR_red_3 = RGB_NIR_CORR[15:20]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_3, split=3, tile=tile)

        RGB_NIR_CORR_red_4 = RGB_NIR_CORR[20:25]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_4, split=4, tile=tile)

        RGB_NIR_CORR_red_5 = RGB_NIR_CORR[25:30]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_5, split=5, tile=tile)

    elif tile in [1, 2, 3]:  # 20 orbits
        RGB_NIR_CORR_red_0 = RGB_NIR_CORR[0:5]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_0, split=0, tile=tile)

        RGB_NIR_CORR_red_1 = RGB_NIR_CORR[5:10]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_1, split=1, tile=tile)

        RGB_NIR_CORR_red_2 = RGB_NIR_CORR[10:15]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_2, split=2, tile=tile)

        RGB_NIR_CORR_red_3 = RGB_NIR_CORR[15:20]
        avg_and_save(RGBNIR=RGB_NIR_CORR_red_3, split=3, tile=tile)

    # print()