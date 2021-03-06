import os
import random
import sys

import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

from ..utils.utilities import (create_folder, get_filename, create_logging,
    load_scalar, get_labels, write_submission_csv)
from 单任务.utils.data_generator import DataGenerator, TestDataGenerator
from models import Cnn_5layers_AvgPooling, Cnn_9layers_MaxPooling, Cnn_9layers_AvgPooling, Cnn_13layers_AvgPooling,\
    Cnn_13layers_MaxPooling
from losses import binary_cross_entropy
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward
import config


NUM_HOURS = 24
NUM_DAYS = 7
NUM_WEEKS = 52


def mixup_data(x1, x2, y, alpha=1.0, use_cuda=True):  # 数据增强，看下那个博客
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # 随机生成一个（1,1）的张量
    else:
        lam = 1
    #
    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # 给定参数n，返回一个从0到n-1的随机整数序列
    else:
        index = torch.randperm(batch_size)  # 使用cpu还是gpu

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]  # 混合数据
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(args):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      holdout_fold: '1' | 'None', where '1' indicates using validation and 
          'None' indicates using full data for training
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    plt_x = []
    plt_y = []
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    max_iteration = 10      # Number of mini-batches to evaluate on training data
    reduce_lr = True

    labels = get_labels(taxonomy_level)
    classes_num = len(labels)

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    train_hdf5_path = os.path.join(workspace, 'features',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'train.h5')

    validate_hdf5_path = os.path.join(workspace, 'features',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'validate.h5')

    scalar_path = os.path.join(workspace, 'scalars',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'train.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_folder(checkpoints_dir)

    _temp_submission_path = os.path.join(workspace, '_temp_submissions', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type, '_submission.csv')
    create_folder(os.path.dirname(_temp_submission_path))

    validate_statistics_path = os.path.join(workspace, 'statistics', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type,
        'validate_statistics.pickle')
    create_folder(os.path.dirname(validate_statistics_path))

    annotation_path = os.path.join(dataset_dir, 'annotations.csv')

    yaml_path = os.path.join(dataset_dir, 'dcase-ust-taxonomy.yaml')

    logs_dir = os.path.join(workspace, 'logs', filename, args.mode,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    if cuda:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Load scalar
    scalar = load_scalar(scalar_path)

    # Model
    Model = eval(model_type)
    model = Model(classes_num)

    if cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    # Data generator
    data_generator = DataGenerator(
        train_hdf5_path=train_hdf5_path,
        validate_hdf5_path=validate_hdf5_path,
        holdout_fold=holdout_fold,
        scalar=scalar,
        batch_size=batch_size)

    # Evaluator
    evaluator = Evaluator(
        model=model,
        data_generator=data_generator,
        taxonomy_level=taxonomy_level,
        cuda=cuda,
        verbose=False)

    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)

    train_bgn_time = time.time()
    iteration = 0
    best_inde = {}
    best_inde['micro_auprc'] = np.array([0.0])
    best_inde['micro_f1'] = np.array([0.0])
    best_inde['macro_auprc'] = np.array([0.0])
    best_inde['average_precision'] = np.array([0.0])
    best_inde['sum'] = best_inde['micro_auprc'] + best_inde['micro_f1'] + best_inde['macro_auprc']
    best_map=0

    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():



        # Evaluate
        if iteration % 200 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}, {} level statistics:'.format(
                iteration, taxonomy_level))

            train_fin_time = time.time()

            # Evaluate on training data
            if mini_data:
                raise Exception('`mini_data` flag must be set to False to use '
                    'the official evaluation tool!')

            train_statistics = evaluator.evaluate(
                data_type='train',
                max_iteration=None)
            if iteration > 5000:
                if best_map<np.mean(train_statistics['average_precision']):
                    best_map=np.mean(train_statistics['average_precision'])
                    logging.info('best_map= {}'.format(best_map))
                    # logging.info('iter= {}'.format(iteration))
                    checkpoint = {
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'indicators': train_statistics}
                    checkpoint_path = os.path.join(
                        checkpoints_dir, 'best2.pth')
                    torch.save(checkpoint, checkpoint_path)
                    logging.info('best_models saved to {}'.format(checkpoint_path))



            # Evaluate on validation data
            if holdout_fold != 'none':
                validate_statistics = evaluator.evaluate(
                    data_type='validate',
                    submission_path=_temp_submission_path,
                    annotation_path=annotation_path,
                    yaml_path=yaml_path,
                    max_iteration=None)

                validate_statistics_container.append_and_dump(
                    iteration, validate_statistics)


            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))


            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9


        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'fine_target', 'coarse_target', 'spacetime']:
                batch_data_dict[key] = move_data_to_gpu(
                    batch_data_dict[key], cuda)

        feature, spacetime, targets_a, targets_b, lam = mixup_data(batch_data_dict['feature'],
                                                                        batch_data_dict['spacetime'],
                                                                        batch_data_dict['{}_target'.format(taxonomy_level)], alpha=1.0,
                                                                        use_cuda=True)


        # Train
        model.train()
        criterion = nn.BCELoss().cuda()
        batch_output = model(feature, spacetime)

        # loss
        #batch_target = batch_data_dict['{}_target'.format(taxonomy_level)]
        loss = mixup_criterion(criterion, batch_output, targets_a, targets_b, lam)
        #loss = binary_cross_entropy(batch_output, batch_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 100 == 0:
            plt_x.append(iteration)
            plt_y.append(loss.item())
        if iteration % 10000 == 0 and iteration != 0:
            plt.figure(1)
            plt.suptitle('test result ', fontsize='18')
            plt.plot(plt_x, plt_y, 'r-', label='loss')
            plt.legend(loc='best')
            plt.savefig('/home/fangjunyan/count/'+time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))+'{}'.format(holdout_fold)+'{}.jpg'.format(taxonomy_level))
        # Stop learning
        if iteration == 10000:
            break

        iteration += 1


def inference_validation(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      iteration: int
      holdout_fold: '1', which means using validation data
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    iteration = args.iteration
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    visualize = args.visualize
    filename = args.filename

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second

    labels = get_labels(taxonomy_level)
    classes_num = len(labels)

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    train_hdf5_path = os.path.join(workspace, 'features',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'train.h5')

    validate_hdf5_path = os.path.join(workspace, 'features',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'validate.h5')

    scalar_path = os.path.join(workspace, 'scalars',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'train.h5')

    checkpoint_path = os.path.join(workspace, 'checkpoints', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type,
        '{}_iterations.pth'.format(iteration))

    submission_path = os.path.join(workspace, 'submissions', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type, 'submission.csv')
    create_folder(os.path.dirname(submission_path))

    annotation_path = os.path.join(dataset_dir, 'annotations.csv')

    yaml_path = os.path.join(dataset_dir, 'dcase-ust-taxonomy.yaml')

    logs_dir = os.path.join(workspace, 'logs', filename, args.mode,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if cuda:
        model.cuda()

    # Data generator
    data_generator = DataGenerator(
        train_hdf5_path=train_hdf5_path,
        validate_hdf5_path=validate_hdf5_path,
        holdout_fold=holdout_fold,
        scalar=scalar,
        batch_size=batch_size)

    # Evaluator
    evaluator = Evaluator(
        model=model,
        data_generator=data_generator,
        taxonomy_level=taxonomy_level,
        cuda=cuda,
        verbose=True)

    # Evaluate on validation data
    evaluator.evaluate(
        data_type='validate',
        submission_path=submission_path,
        annotation_path=annotation_path,
        yaml_path=yaml_path,
        max_iteration=None)

    # Visualize
    if visualize:
        evaluator.visualize(data_type='validate')


def inference_evaluation(args):
    '''Inference on evaluation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      taxonomy_level: 'fine' | 'coarse'
      model_type: string, e.g. 'Cnn_9layers_MaxPooling'
      iteration: int
      holdout_fold: 'none', which means using model trained on all development data
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    taxonomy_level = args.taxonomy_level
    model_type = args.model_type
    iteration = args.iteration
    holdout_fold = args.holdout_fold
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename

    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second

    labels = get_labels(taxonomy_level)
    classes_num = len(labels)

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    evaluate_hdf5_path = os.path.join(workspace, 'features',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'evaluate.h5')

    scalar_path = os.path.join(workspace, 'scalars',
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'train.h5')

    checkpoint_path = os.path.join(workspace, 'checkpoints', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type,
        'best2.pth')

    submission_path = os.path.join(workspace, 'submissions', filename,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type, 'best2_submission.csv')
    create_folder(os.path.dirname(submission_path))

    logs_dir = os.path.join(workspace, 'logs', filename, args.mode,
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'taxonomy_level={}'.format(taxonomy_level),
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)

    # Load model
    Model = eval(model_type)
    model = Model(classes_num)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    if cuda:
        model.cuda()

    # Data generator
    data_generator = TestDataGenerator(
        hdf5_path=evaluate_hdf5_path,
        scalar=scalar,
        batch_size=batch_size)

    # Forward
    output_dict = forward(
        model=model,
        generate_func=data_generator.generate(),
        cuda=cuda,
        return_target=False)

    # Write submission
    write_submission_csv(
    audio_names=output_dict['audio_name'],
    outputs=output_dict['output'],
    taxonomy_level=taxonomy_level,
    submission_path=submission_path)

def seed_everything(seed):#设定随机数2020
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False, help='Visualize log mel spectrogram of different sound classes.')
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    parser_inference_evaluation = subparsers.add_parser('inference_evaluation')
    parser_inference_evaluation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_evaluation.add_argument('--workspace', type=str, required=True)
    parser_inference_evaluation.add_argument('--taxonomy_level', type=str, choices=['fine', 'coarse'], required=True)
    parser_inference_evaluation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_evaluation.add_argument('--holdout_fold', type=str, choices=['none'], required=True)
    parser_inference_evaluation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_evaluation.add_argument('--batch_size', type=int, required=True)
    parser_inference_evaluation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_evaluation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        SEED = 2020
        seed_everything(SEED)
        train(args)

    elif args.mode == 'inference_validation':
        SEED = 2020
        seed_everything(SEED)
        inference_validation(args)

    elif args.mode == 'inference_evaluation':
        SEED = 2020
        seed_everything(SEED)
        inference_evaluation(args)

    else:
        raise Exception('Error argument!')
