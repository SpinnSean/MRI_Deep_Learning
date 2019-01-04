import pandas as pd
import numpy as np
import nibabel as nib
from create_data_df import create_data_df
import os
from helpers import nii2Numpy, normalize


def attribute_category(imagesDf, category, labelName, ratio, verbose=1):
    ''' This function distributes each subject in a 'train' or 'test' category.
    Args:
        imagesDf (pd.DataFrame): a pd.DataFrame that contains the info of all files
            by subject.
        ratios (list): a list containing the proportions of train/test
            subjects. should sum to 1 and supposedly it has been tested before.
    Returns:
        imagesDf (pd.DataFrame): a pd.DataFrame that contains the info of all files
            by subject where the 'category' column has been set to either
            train or test depending the result of the random draw.
            The value of test or train is the same for a given subject.
    '''
    nImages = imagesDf.shape[0]
    n = int(round(nImages * ratio))
    i = 0

    category_classes = pd.Series(imagesDf['labels'])
    unique_category_classes = category_classes.unique()


    while True:
        for r in unique_category_classes:
            unknown_imagesDf = imagesDf[(imagesDf.category == "unknown") & (category_classes == r)]
            n_unknown = unknown_imagesDf.shape[0]
            if n_unknown == 0: continue
            random_i = np.random.randint(0, n_unknown)
            row = imagesDf[imagesDf.category == "unknown"].iloc[random_i,]
            imagesDf.loc[imagesDf.index[imagesDf.subjects == row.subjects], 'category'] = category
            i += imagesDf.loc[imagesDf.index[imagesDf.subjects == row.subjects], 'category'].shape[0]
            if i >= n: break

        n_unknown = imagesDf[(imagesDf.category == "unknown") & (category_classes == r)].shape[0]
        if i >= n or n_unknown == 0: break

    if verbose > 0:
        print(category, ": expected/real ratio = %3.2f / %3.2f" % (
        100. * ratio, 100. * imagesDf.category.loc[imagesDf.category == category].shape[0] / nImages))


def count_valid_samples(imagesDf):
    
    total_slices = 0
    imagesDf['valid_samples'] = np.repeat(0, imagesDf.shape[0])
    imagesDf['total_samples'] = np.repeat(0, imagesDf.shape[0])
    for index, row in imagesDf.iterrows():
        subMRI = nii2Numpy(imagesDf.paths)

        subMRI = normalize(subMRI)

        imagesDf['total_samples'].iloc[index] = subMRI.shape[0]
        valid_slices = 0
        for j in range(subMRI.shape[0]):
            if subMRI[j].sum() != 0:
                valid_slices += 1
        imagesDf['valid_samples'].iloc[index] = valid_slices
        total_slices += valid_slices
    return (total_slices)



def prepare_data(mainDir, data_dir, report_dir, input_str, ext, labelName, idColumn, covPath, ratios=[0.75,0.15], batch_size=2, feature_dim=2, pad_base=0):
    # data = {}
    # # Setup file names and imagesDfput directories
    # data["train_x_fn"] = os.path.join(data_dir,'train_x')
    #
    # data["train_y_fn"] = os.path.join(data_dir,'train_y')
    # data["validate_x_fn"] = os.path.join(data_dir,'validate_x')
    #
    # data["validate_y_fn"] = os.path.join(data_dir,'validate_y')
    # data["test_x_fn"] = os.path.join(data_dir,'test_x')
    #
    # data["test_y_fn"] = os.path.join(data_dir,'test_y')

    #if images_fn == None: images_fn = os.path.join(report_dir,'images.csv')

    imagesDf = create_data_df(mainDir, input_str, ext, labelName, covPath, idColumn)

    # fix to make sure enough test/train/validate
    attribute_category(imagesDf, 'train', 'labels', ratios[0])
    attribute_category(imagesDf, 'validate','labels', ratios[1])
    imagesDf.category.loc[ imagesDf.category == "unknown" ] = "test"


    fname = os.path.join(mainDir, 'sMRI_{}.csv'.format(labelName))
    imagesDf.to_csv(fname, sep=',')

    return imagesDf


def unitTest1():

    mainDir = 'maindir/'

    # Create fake dataframe as it should be done by create_data_imagesDf
    n=50
    subjects=[]
    paths=[]
    labels=[]
    i=1

    while(i<n):
        if np.random.random() > 0.5:
           labels.append('Male')
        else:
            labels.append('Female')

        subjects.append('sub-{:03d}'.format(i))
        paths.append(os.path.join(mainDir, 'sub-{:03d}'.format(i), 'sub-{:03d}_T1w.mnc'.format(i)))
        i+=1

    imagesDf=pd.DataFrame({'subjects': subjects,
                     'paths': paths,
                     'labels': labels})

    imagesDf["category"] = "unknown"

    return imagesDf
