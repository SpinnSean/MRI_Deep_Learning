import pandas as pd
from pathlib import Path
import numpy as np
import glob
import os
import re


def extractSubjName(path):
    p=Path(path)
    subject = [pp for pp in p.parts if 'sub' in pp]
    #if len(subject) > 1:
     #   print("Duplicate participant for " + subject[0])
     #   return -1
    return subject[0]

# To Do: somehow specify what ids look like, if they have text or not, leading 0, etc.
def renameSubject(code):
    if code == len(code)*' ':
        return ''
    #num = re.findall(r'_\d+', code)[0][-3:]
    return 'sub-' + code


def create_labels(labelName, mainDir, covPath, idColumn):
    """2
    Extracts the covariate which is used as label
    and returns a df with covariate and part name as
    columns.

    :param labelName:
    :param covPath:
    :return: pandas df and label dictionary
    """
    try:
        covDf = pd.read_csv(os.path.join(mainDir,covPath),
                            usecols=[idColumn,labelName],
                            converters={idColumn: lambda x: str(x)})

    except ValueError:
        covDf = pd.read_csv(os.path.join(mainDir,covPath),
                            usecols=[idColumn,labelName],
                            sep=';',
                            converters={idColumn: lambda x: str(x)},
                            #dtype='str'
                            )


    labels = covDf[labelName].dropna().unique()

    if len(labels) > 2:
        print("This is for binary classification. Too many classes.")
        return -1
    labelMapDict = {labels[0]: 0,
                    labels[1]: 1}

    # Replace labels with 0 and 1
    covDf[labelName].replace(labelMapDict,inplace=True)
    # Remove any NaNs, row-wise
    covDf.dropna(0, how='any', inplace=True)

    # Removing whitespaces, replacing empty strings with NaNs
    covDf[idColumn] = covDf[idColumn].str.strip()
    covDf = covDf.replace({r'\s+':np.nan, '': np.nan}, regex=True)

    return [covDf, labelMapDict]



def create_data_df(mainDir,data_dir, input_str, ext, labelName, covPath, idColumn):
    """
    Creates a dataframe with the participants filepath row by row.
     Assumes BIDS format.

    :param mainDir: directory to participant directories
    :param fstr: substring to find the image file (ex: T1w)
    :param ext: extension for the image file (ex: "mnc")
    :param labelName: name of variable to use in covariate file
    :param covPath: name of csv covariate file (ex: COMPLETE_NEUROVENTURE.csv)
    :return: pandas df with file names as rows
    """

    imgFiles = glob.glob(os.path.join(mainDir,"sub*",data_dir,"*"+input_str+"*"+ext))
    dataDf = pd.DataFrame({'paths': imgFiles})
    dataDf[idColumn] = dataDf.paths.map(extractSubjName)

    [labelsDf, labelDict] = create_labels(labelName, mainDir, covPath, idColumn)
    labelsDf.dropna(0, how='any', inplace=True)
    labelsDf[idColumn] = labelsDf[idColumn].map(renameSubject)

    dataDf = pd.merge(dataDf,labelsDf,on=[idColumn])
    dataDf = dataDf[[idColumn, labelName, 'paths']]
    dataDf.rename(columns={idColumn: 'subjects', labelName: 'labels'}, inplace=True)
    dataDf['category'] = "unknown"
    dataDf.reset_index(inplace=True)
    pd.DataFrame(labelDict, [0]).to_csv(os.path.join(mainDir,'label_dictionary.txt'), index= False)

    return dataDf


def unitTest1():
    covDf = pd.DataFrame({'subjects': ['sub-01', 'sub-02'], 'LabelName': ['Male', 'Female']})