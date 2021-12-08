import h5py
import numpy

TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 1-TRAIN_PERCENTAGE

PATH_OLD = '../data/imgint_trainset_v2.hdf5'
PATH_NEW_TRAIN = '../data/imgint_trainset_v3.hdf5'
PATH_NEW_VAL = '../data/imgint_validationset_v3.hdf5'

file_old = h5py.File(PATH_OLD, 'r')
# for checking (later) that dimensions match (in new dataset)
data_old = file_old['data']
gt_old = file_old['gt']

N_SAMPLES = data_old.shape[0]
N_SAMPLES_TRAIN = int(TRAIN_PERCENTAGE*N_SAMPLES)

# get data for new train dataset
data_new_train = file_old['data'][0:N_SAMPLES_TRAIN, :, :]
gt_new_train = file_old['gt'][0:N_SAMPLES_TRAIN]

# residual values go to test dataset
data_new_val = file_old['data'][N_SAMPLES_TRAIN:, :, :]
gt_new_val = file_old['gt'][N_SAMPLES_TRAIN:]


with h5py.File(PATH_NEW_TRAIN, "w") as f1:
    dset1_1 = f1.create_dataset('data', data=data_new_train)
    dset1_2 = f1.create_dataset('gt', data=gt_new_train)

with h5py.File(PATH_NEW_VAL, "w") as f2:
    dset2_1 = f2.create_dataset('data', data=data_new_val)
    dset2_2 = f2.create_dataset('gt', data=gt_new_val)