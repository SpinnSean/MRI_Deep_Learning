from prepare_data import  *
from create_PNG_dataset import create_PNG_dataset
import os

def mri_keras(mainDir, data_dir, report_dir, input_str,  ext, labelName, covPath, ratios=[0.75,0.15], batch_size=2, feature_dim=2, pad_base=0, createDataset=True):

    #setup_dirs(target_dir)

    imagesDf = prepare_data(mainDir, data_dir, report_dir, input_str, ext, labelName, covPath, ratios, batch_size, feature_dim, pad_base)

    if createDataset:
        create_PNG_dataset(imagesDf.paths)
