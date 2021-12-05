import h5py
import numpy

PATH_OLD = '../data/imgint_trainset_v2.hdf5'
PATH_NEW = '../data/imgint_trainset_v2_reduced.hdf5'

file_old = h5py.File(PATH_OLD, 'r')
# for checking (later) that dimensions match (in new dataset)
data_old = file_old['data']
gt_old = file_old['gt']

data_new = file_old['data'][0:500, :, :]
gt_new = file_old['gt'][0:500]

with h5py.File(PATH_NEW, "w") as f:
    dset1 = f.create_dataset('data', data=data_new)
    dset2 = f.create_dataset('gt', data=gt_new)

