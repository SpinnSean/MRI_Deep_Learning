import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from prepare_data import  *
from helpers import *
from create_PNG_dataset import create_PNG_dataset
from keras_models import *
import tensorflow as tf
from plotting import plotLoss, comparePredictions
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



def mri_keras(main_dir, data_dir, report_dir, target_dir, input_str,  ext, labelName, idColumn, covPath, ratios=[0.75,0.15], createPNGDataset=False, batch_size=2, nb_epoch=10, images_to_predict=None, clobber=False, model_name='model.hdf5',model_type='cnn-autoencoder', images_fn='images.csv',nK="16,32,64,128", n_dil=None, kernel_size=64, drop_out=0, loss='mse', activation_hidden="relu", activation_output="sigmoid", metric="categorical_accuracy", pad_base=0,  verbose=0, make_model_only=False,nGPU=1):

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
        if nGPU == 0:
            print("[INFO] training with 1 CPU...")
        else:
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


    #memRequired = get_model_memory_usage(batch_size, model)
    #if memRequired > 12.0:
    #    print("Required memory: {}\nAvailable memory: {}\nTry reducing the batch size.".format(memRequired,12.0))
     #   exit(0)


    ### 2) Train network on data
    model_fn = set_model_name(model_name, model_dir)
    history_fn = str(os.path.join(target_dir, 'model', str(os.path.basename(model_fn).split('.')[0]) + '_history.json'))

    # Load cropped and padded images
    X_train = np.load(data["x_train_fn"] + '.npy')
    Y_train = np.load(data["y_train_fn"] + '.npy')
    X_validate = np.load(data["x_validate_fn"] + '.npy')
    X_test = np.load(data["x_test_fn"] + '.npy')
    Y_test = np.load(data["y_test_fn"] + '.npy')


    print('Model:', model_fn)
    if not os.path.exists(model_fn) or clobber:
        # If model_fn does not exist, or user wishes to write over (clobber) existing model
        # then train a new model and save it

        # Modify data for model architecture
        X_train, X_validate, X_test,  Y_train, Y_validate, Y_test = dataConfiguration(X_train,
                                                                                    X_validate,
                                                                                    X_test,
                                                                                     Y_train,
                                                                                    Y_validate,
                                                                                    Y_test,
                                                                                    model_type,
                                                                                    data["image_dim"],
                                                                                    nlabels)

        model, _ = compile_and_run(target_dir,
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
                                         verbose=verbose,
                                         nGPU=nGPU)

        print("Fitting over.")
    else:
        # Load model
        model = load_model(model_fn)
        with open(history_fn) as f:
            history = json.load(f)
    print("Loss plot create in ./plots/")
    with open(history_fn) as f:
        history = json.load(f)

    plotLoss(history, nb_epoch, model_name[:-5])

    X_test_mid = X_test[extractMiddleSlices(X_test)]
    X_validate_mid = X_validate[extractMiddleSlices(X_validate)]
    predVal = model.predict(X_validate_mid)
    predTest = model.predict(X_test_mid)
    comparePredictions(X_validate_mid,predVal,'validate',model_name[:-5])
    comparePredictions(X_test_mid, predTest, 'test', model_name[:-5])
    print("Predictions complete. A comparison plot is stored in ./plots for validation and test sets.")



if __name__ == '__main__':
    params = readParameters('mri_keras_args')
    mri_keras(**params)

