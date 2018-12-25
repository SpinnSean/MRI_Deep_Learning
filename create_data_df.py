import pandas as pd
from pathlib import Path
import glob
import os


def extractSubjName(path):
    p=Path(path)
    subject = [pp for pp in p.parts if 'sub' in pp]
    if len(subject) > 1:
        print("Duplicate participant for " + subject[0])
        return -1
    return subject[0]


def create_labels(labelName,covPath):
    """
    Extracts the covariate which is used as label
    and returns a df with covariate and part name as
    columns.

    :param labelName:
    :param covPath:
    :return: pandas df and label dictionary
    """
    covDf = pd.read_csv(covPath, usecols=['subjects',labelName])
    labels = covDf[labelName].unique()

    if  len(labels) > 2:
        print("This is for binary classification. Too many classes.")
        return -1
    labelMapDict = {labels[0]: 0,
                    labels[1]: 1}

    # Replace labels with 0 and 1
    covDf[labelName].replace(labelMapDict,inplace=True)
    covDf.rename(index=str, columns={labelName: 'labels'}, inplace=True)

    return {'labelsDf':covDf, 'dict': labelMapDict}



def create_data_df(mainDir, input_str, ext, labelName, covPath):
    """
    Creates a dataframe with the participants filepath row by row.
     Assumes BIDS format.

    :param mainDir: directory to participant directories
    :param fstr: substring to find the image file (ex: *T1w*)
    :param ext: extension for the image file (ex: "mnc")
    :param labelName: name of variable to use in covariate file
    :param covPath: name of csv covariate file (ex: COMPLETE_NEUROVENTURE.csv)
    :return: pandas df with file names as rows
    """

    imgFiles = glob.glob(os.path.join(mainDir,"*","*"+input_str+"*"+ext))
    dataDf = pd.DataFrame({'paths': imgFiles})
    dataDf['subjects'] = dataDf.paths.map(extractSubjName)

    [labelsDf, labelDict] = create_labels(labelName, covPath)

    dataDf = pd.merge(dataDf,labelsDf,on='subjects')
    dataDf = dataDf[['subjects', 'labels']]
    dataDf['category'] = "unknown"
    dataDf.reset_index(inplace=True)

    return dataDf


def unitTest1():
    covDf = pd.DataFrame({'subjects': ['sub-01', 'sub-02'], 'LabelName': ['Male', 'Female']})