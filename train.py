from __future__ import print_function
import os
from data_generator import   MY_Generator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import config as cfg
from model import model1 ,modeltest
from prepare_data import create_folder ,load_csv



def train():
    num_classes = cfg.num_classes
    n_time = cfg.n_time
    n_freq = cfg.n_freq

    # Load training & testing data
    trdata = load_csv( 'feature /train.csv')
    valdata= load_csv( 'feature /val.csv')


    # Build model
    model = modeltest(n_time, n_freq,  num_classes)

    # Compile model
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    model.summary()
    # Save model callback
    filepath = os.path.join(cfg.out_model_dir, "gatedAct_AudioAge.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)

    # Data generator
    My_gen = MY_Generator( filenames= trdata , batch_size= 20)

    # get1= My_gen.get1(1)
    # Train
    model.fit_generator(generator=My_gen ,
                        # validation_data=(te_x, te_y),
                        steps_per_epoch= 10,  # num_training_samples // batch_size
                        epochs= 200 ,  # Maximum 'epoch' to train
                        verbose= 1 ,
                        callbacks=[save_model])




