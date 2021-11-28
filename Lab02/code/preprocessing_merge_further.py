import numpy as np
import matplotlib.pyplot as plt
import h5py

def avg_and_save(RGBNIR, tile):
    averaged_tile = np.nanmean(RGBNIR, axis=0)
    print(f'saving final version of tile {tile}')
    fpath = f'C:\scratch\RGBNIR_final_tile_{tile}'
    np.save(fpath, averaged_tile)
    print(f'final version of tile {tile} saved sucessfully')

TILES = [0, 1, 2, 3, 4, 5]
SPLIT = 5

for tile in TILES:
    FILEPATH_BASE_TILE = f'C:\scratch\RGBNIR_red_tile_{tile}'  # windows

    # load splits

    if tile in [0, 4, 5]:  # 30 orbits
        RGBNIR_split0 = np.load(f'{FILEPATH_BASE_TILE}_split{0}.npy')
        RGBNIR_split1 = np.load(f'{FILEPATH_BASE_TILE}_split{1}.npy')
        RGBNIR_split2 = np.load(f'{FILEPATH_BASE_TILE}_split{2}.npy')
        RGBNIR_split3 = np.load(f'{FILEPATH_BASE_TILE}_split{3}.npy')
        RGBNIR_split4 = np.load(f'{FILEPATH_BASE_TILE}_split{4}.npy')
        RGBNIR_split5 = np.load(f'{FILEPATH_BASE_TILE}_split{5}.npy')

        RGBNIR_stacked = np.stack((RGBNIR_split0, RGBNIR_split1, RGBNIR_split2, RGBNIR_split3, RGBNIR_split4, RGBNIR_split5), axis=0)

    elif tile in [1, 2, 3]:  # 20 orbits
        RGBNIR_split0 = np.load(f'{FILEPATH_BASE_TILE}_split{0}.npy')
        RGBNIR_split1 = np.load(f'{FILEPATH_BASE_TILE}_split{1}.npy')
        RGBNIR_split2 = np.load(f'{FILEPATH_BASE_TILE}_split{2}.npy')
        RGBNIR_split3 = np.load(f'{FILEPATH_BASE_TILE}_split{3}.npy')

        RGBNIR_stacked = np.stack((RGBNIR_split0, RGBNIR_split1, RGBNIR_split2, RGBNIR_split3, RGBNIR_split4, RGBNIR_split5), axis=0)


    avg_and_save(RGBNIR_stacked, tile)
