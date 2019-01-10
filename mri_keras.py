import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from prepare_data import  *
from helpers import *
from create_PNG_dataset import create_PNG_dataset
from keras_models import *
import tensorflow as tf
from keras import backend
#from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
import argparse
from sys import argv, exit
from glob import glob

session_conf = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

def create_dir_verbose(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directory:", directory)
    return directory


def setup_dirs(target_dir="./"):
    global data_dir
    global report_dir
    global train_dir
    global test_dir
    global validate_dir
    global model_dir
    data_dir = target_dir + os.sep + 'data' + os.sep
    report_dir = target_dir + os.sep + 'report' + os.sep
    train_dir = target_dir + os.sep + 'predict' + os.sep + 'train' + os.sep
    test_dir = target_dir + os.sep + 'predict' + os.sep + 'test' + os.sep
    validate_dir = target_dir + os.sep + 'predict' + os.sep + 'validate' + os.sep
    model_dir = target_dir + os.sep + 'model'
    create_dir_verbose(train_dir)
    create_dir_verbose(test_dir)
    create_dir_verbose(validate_dir)
    create_dir_verbose(data_dir)
    create_dir_verbose(report_dir)
    create_dir_verbose(model_dir)
    return 0


def mri_keras(main_dir, data_dir, report_dir, target_dir, input_str,  ext, labelName, idColumn, covPath, ratios=[0.75,0.15], createPNGDataset=False, batch_size=2, nb_epoch=10, images_to_predict=None, clobber=False, model_fn='model.hdf5',model_type='cnn-autoencoder', images_fn='images.csv',nK="16,32,64,128", n_dil=None, kernel_size=64, drop_out=0, loss='mse', activation_hidden="relu", activation_output="sigmoid", metric="categorical_accuracy", pad_base=0,  verbose=1, make_model_only=False,nGPU=1):

    setup_dirs(target_dir)
    #PNG_DIM=[256,256]
    #NUM_IMG= 256

    imagesDfOut = os.path.join(main_dir,'imagesDf_'+labelName+'.csv')
    [imagesDf,data] = prepare_data(main_dir, data_dir, report_dir, input_str, ext, labelName, idColumn, covPath, imagesDfOut, ratios)


     #if createPNGDataset:
        #NUM_IMG, PNG_DIM = create_PNG_dataset(imagesDf.paths.tolist())
        #data = load_data(main_dir, data_dir, imagesDf, PNG_DIM, NUM_IMG )

    ### 1) Define architecture of neural network
    Y_validate=np.load(data["y_validate_fn"]+'.npy')
    nlabels=len(np.unique(Y_validate)) #Number of unique labels in the labeled images

    if nGPU <= 1:
        print("[INFO] training with 1 GPU...")
        model = build_model(data["image_dim"], nlabels, nK, n_dil, kernel_size, drop_out, model_type=model_type,
                            activation_hidden=activation_hidden, activation_output=activation_output, loss=loss,
                            verbose=0)
        if make_model_only: return 0
    else:
        print("[INFO] training with {} GPUs...".format(nGPU))

        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = build_model(data["image_dim"], nlabels, nK, n_dil, kernel_size, drop_out, model_type=model_type,
                                activation_hidden=activation_hidden, activation_output=activation_output, loss=loss,
                                verbose=0)
        model = multi_gpu_model(model, gpus=nGPU)


    memRequired = get_model_memory_usage(batch_size, model)
    if memRequired > 12.0:
        print("Required memory: {}\nAvailable memory: {}\nTry reducing the batch size.".format(memRequired,12.0))
        exit(0)


    ### 2) Train network on data
    model_fn = set_model_name(model_fn, model_dir)
    history_fn = str(os.path.join(target_dir, 'model', str(os.path.basename(model_fn).split('.')[0]) + '_history.json'))

    print('Model:', model_fn)
    if not os.path.exists(model_fn):
        # If model_fn does not exist, or user wishes to write over (clobber) existing model
        # then train a new model and save it

        # Load cropped and padded images
        X_train = np.load(data["x_train_fn"] + '.npy')
        Y_train = np.load(data["y_train_fn"] + '.npy')
        X_validate = np.load(data["x_validate_fn"] + '.npy')
        X_test = np.load(data["x_test_fn"] + '.npy')
        Y_test = np.load(data["y_test_fn"] + '.npy')


        # Modify data for model architecture
        X_train, X_validate, X_test = dataConfiguration(X_train,
                                                        X_validate,
                                                        X_test,
                                                        model_type,
                                                        data["image_dim"])

        model, history = compile_and_run(target_dir,
                                         model,
                                         model_fn,
                                         model_type,
                                         history_fn,
                                         X_train,
                                         Y_train,
                                         X_validate,
                                         Y_validate,
                                         nb_epoch,
                                         batch_size,
                                         nlabels,
                                         loss=loss,
                                         verbose=1,
                                         nGPU=nGPU)

        print("Fitting over.")
    else:
        # Load model
        model = model.load_weights(model_fn)

    #pred = model.predict(X_validate[extractMostInfSlice(X_validate)])


def unitTest1():

    mri_keras('/data/IMAGEN/BIDS/derivatives/BIDS/FU2', 'anat', './reports', './processed','T1w', 'nii', 'gender', 'ID', 'IMAGEN_Test.csv',ratios=[0.75,0.15], createPNGDataset=False, batch_size=256, nb_epoch=50, images_to_predict=None, clobber=False, model_fn='autoencoder2.hdf5')



if __name__ == '__main__':
    params = readParameters('mri_keras_args')
    mri_keras(**params)

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
