#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from yolo import create_yolov3_model, MaskedConv2D
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import time
from keras import backend as K
import random
from train import create_training_instances
from yolo import dummy_loss



def cal_similar(t1, t2):
    cos_sim = np.sum(t1 * t2) / np.linalg.norm(t1) * np.linalg.norm(t2)
    dis_sim = np.sqrt(np.sum(np.square(t1 - t2)))
    return (1-cos_sim) * dis_sim

def print_nonzeros(model):
    nonzeros = total = 0
    for i in range(1, 252):
        layer = model.get_layer(index=i)
        if 'conv' in layer.name:
            weights = np.array(layer.get_weights()[0])
            nz_count = np.count_nonzero(weights)
            total_params = np.prod(weights.shape)
            nonzeros += nz_count
            total += total_params
            print(layer.name)
            print("Before Prunning, there are", total_params, "parameters")
            print("After Prunning, there are", nz_count, "parameters")

    print("The total model gets", total, "prunned",total - nonzeros, " while having", nonzeros, "left")

def _main_(args):
    config_path = args.conf
    prune = args.prune

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################
    valid_ints, labels = parse_voc_annotation(
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)

    valid_generator = BatchGenerator(
        instances           = valid_ints,
        anchors             = config['model']['anchors'],
        labels              = labels,
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],
        shuffle             = True,
        jitter              = 0.0,
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    infer_model = load_model(config['train']['saved_weights_name'], custom_objects={'MaskedConv2D':MaskedConv2D})
    infer_model.load_weights('cluster.h5')
    #infer_model.load_weights("Pruned.h5")
    prune = int(prune)
    if prune == 2:
        print("Do Percentile-Prunning since prunning factor:", prune)
        q = 30
        biasFlag = False
        for i in range(1, 252):
            layer = infer_model.get_layer(index=i)
            if 'conv' in layer.name:
                layer.prune_by_threshold(q)
        print_nonzeros(infer_model)
    elif prune == 3:
        print("Do XJB-Prunning since prunning factor:", prune)
        infer_model.summary()
        for i in range(1, 252):
            layer = infer_model.get_layer(index=i)
            if 'conv' in layer.name:
                layer.prune_by_similarity()

        print_nonzeros(infer_model)

    print("Before Finetuning")
    start = time.time()
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    end = time.time()
    print("total time consuming: ", end - start)
    infer_model.save_weights("Pruned.h5", overwrite=True)

    # print("Finetuning")
     # train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
     #     config['train']['train_annot_folder'],
     #     config['train']['train_image_folder'],
     #     config['train']['cache_name'],
     #     config['valid']['valid_annot_folder'],
     #     config['valid']['valid_image_folder'],
     #     config['valid']['cache_name'],
     #     config['model']['labels']
     # )
     #
     #
     # finetune_generator = BatchGenerator(
     #     instances           = train_ints,
     #     anchors             = config['model']['anchors'],
     #     labels              = labels,
     #     downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
     #     max_box_per_image   = max_box_per_image,
     #     batch_size          = config['train']['batch_size'],
     #     min_net_size        = config['model']['min_input_size'],
     #     max_net_size        = config['model']['max_input_size'],
     #     shuffle             = True,
     #     jitter              = 0.3,
     #     norm                = normalize
     # )
     #
     # infer_model.compile(loss=dummy_loss, optimizer='adam')
     # infer_model.fit_generator(
     #         generator        = finetune_generator,
     #         steps_per_epoch  = len(finetune_generator) * config['train']['train_times'],
     #         epochs           = 1,
     #         verbose          = 2 if config['train']['debug'] else 1,
     #         # callbacks        = callbacks,
     #         workers          = 4,
     #         max_queue_size   = 1,
     #         pickle_safe      = False
     # )

    # start = time.time()
    # # compute mAP for all the classes
    # average_precisions = evaluate(infer_model, valid_generator)
    #
    # # print the score
    # for label, average_precision in average_precisions.items():
    #     print(labels[label] + ': {:.4f}'.format(average_precision))
    # print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
    #
    # end = time.time()
    # print("total time consuming: ", end - start)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-p', '--prune', help='whether do prunning or not')
    args = argparser.parse_args()

    _main_(args)
