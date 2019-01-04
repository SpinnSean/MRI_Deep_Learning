from prepare_data import  *
from helpers import load_data
from create_PNG_dataset import create_PNG_dataset
#from keras_models import L1Autoencoder
import os
import argparse
from sys import argv, exit



def mri_keras(main_dir, data_dir, report_dir, input_str,  ext, labelName, idColumn, covPath, ratios=[0.75,0.15], batch_size=2, feature_dim=2, pad_base=0, createDataset=True):

    #setup_dirs(target_dir)
    #global PNG_DIM
    #global NUM_IMG

    imagesDf = prepare_data(main_dir, data_dir, report_dir, input_str, ext, labelName, idColumn, covPath, ratios, batch_size, feature_dim, pad_base)

    if createDataset:
       NUM_IMG, PNG_DIM = create_PNG_dataset(imagesDf.paths.tolist())

    data = load_data(main_dir,imagesDf, PNG_DIM, NUM_IMG)
    print("Dataset Loaded.\nDimensions:")
    print(*["{} : {}".format(k, v.shape) for k, v in data.items()], sep="\n")
    #autoencoder = L1Autoencoder(encDim=25, inShape=125, l1Reg=0.05, loss='mean_squared_error')



def unitTest1():

    mri_keras('/Volumes/Storage/Work/Data/Neuroventure', '' , 'reports', 'brainmask', 'nii', 'DEM_01_Y1', 'Baseline', 'COMPLETE_NEUROVENTURE.csv')


unitTest1()
#if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='size of batch')
    # parser.add_argument('--source', dest='main_dir', required=True, type=str, help='source directory')
    # parser.add_argument('--target', dest='target_dir', required=True, type=str, default="results",
    #                     help='target directory for output (Default: results)')
    # parser.add_argument('--epochs', dest='nb_epoch', type=int, default=10, help='number of training epochs')
    # parser.add_argument('--pad', dest='pad', type=int, default=0,
    #                     help='Images must be divisible by 2^<pad>. Default = 0 ')
    # parser.add_argument('--loss', dest='loss', type=str, default='categorical_crossentropy',
    #                     help='Loss function to optimize network')
    # parser.add_argument('--nK', dest='nK', type=str, default='16,32,64,128', help='number of kernels')
    # parser.add_argument('--n_dil', dest='n_dil', type=str, default=None, help='number of dilations')
    # parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=3, help='Size of kernels')
    # parser.add_argument('--drop-out', dest='drop_out', type=float, default=0.0, help='Drop out rate')
    # parser.add_argument('--metric', dest='metric', type=str, default='categorical_accuracy',
    #                     help='Categorical accuracy')
    # parser.add_argument('--activation-output', dest='activation_output', type=str, default='softmax',
    #                     help='Activation function for last layer of network')
    # parser.add_argument('--activation-hidden', dest='activation_hidden', type=str, default='relu',
    #                     help='Activation function for core convolutional layers of network')
    #
    # # parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=2, help='Warning: option temporaily deactivated. Do not use. Format of features to use (3=Volume, 2=Slice, 1=profile')
    # parser.add_argument('--model', dest='model_fn', default='model.hdf5',
    #                     help='model file where network weights will be saved/loaded. will be automatically generated if not provided by user')
    # parser.add_argument('--model-type', dest='model_type', default='model_0_0',
    #                     help='Name of network architecture to use (Default=model_0_0): unet, model_0_0 (simple convolution-only network), dil (same as model_0_0 but with dilations).')
    # parser.add_argument('--ratios', dest='ratios', nargs=2, type=float, default=[0.7, 0.15, 0.15],
    #                     help='List of ratios for training, validating, and testing (default = 0.7 0.15 0.15)')
    # parser.add_argument('--predict', dest='images_to_predict', type=str, default=None,
    #                     help='either 1) \'all\' to predict all images OR a comma separated list of index numbers of images on which to perform prediction (by default perform none). example \'1,4,10\' ')
    # parser.add_argument('--input-str', dest='input_str', type=str, default='pet', help='String for input (X) images')
    # parser.add_argument('--label-str', dest='label_str', type=str, default='brainmask',
    #                     help='String for label (Y) images')
    # parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='clobber')
    # parser.add_argument('--make-model-only', dest='make_model_only', action='store_true', default=False,
    #                     help='Only build model and exit.')
    # parser.add_argument('-v', '--verbose', dest='verbose', type=int, default=1,
    #                     help='Level of verbosity (0=silent, 1=basic (default), 2=detailed, 3=debug')
    #
    # args = parser.parse_args()
    # args.feature_dim = 2
    #
    # mri_keras(args.main_dir, args.target_dir, input_str=args.input_str, label_str=args.label_str, ratios=args.ratios,
    #            batch_size=args.batch_size, nb_epoch=args.nb_epoch, model_fn=args.model_fn,
    #            model_type=args.model_type, images_to_predict=args.images_to_predict, loss=args.loss, nK=args.nK,
    #            n_dil=args.n_dil, kernel_size=args.kernel_size, drop_out=args.drop_out,
    #            activation_hidden=args.activation_hidden, activation_output=args.activation_output, metric=args.metric,
    #            pad_base=args.pad, verbose=args.verbose, make_model_only=args.make_model_only)
