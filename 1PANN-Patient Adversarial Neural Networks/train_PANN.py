import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data.data_utils import *
from data.dataloader_detection import load_dataset_detection
from data.dataloader_classification import load_pann_classification
from constants import *
from args import get_args
from collections import OrderedDict
from json import dumps
from model.pann import DCRNNModel_pann_classification
from tensorboardX import SummaryWriter
from tqdm import tqdm
#import dotted_dict
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import random



#定义训练操作
def main(args):


    # Get device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)


    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    # Build dataset
    log.info('Building dataset...')
    dataloaders, _, scaler, patient_number = load_pann_classification(
        input_dir=args.input_dir,
        raw_data_dir=args.raw_data_dir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        time_step_size=args.time_step_size,
        max_seq_len=args.max_seq_len,
        standardize=True,
        num_workers=args.num_workers,
        padding_val=0.,
        augmentation=args.data_augment,
        adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
        graph_type=args.graph_type,
        top_k=args.top_k,
        filter_type=args.filter_type,
        use_fft=args.use_fft,
        preproc_dir=args.preproc_dir)

    # Build model
    log.info('Building model...')
    model = DCRNNModel_pann_classification(args=args, num_classes=args.num_classes, num_patient=patient_number, device=device)
    if args.do_train:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model = utils.load_model_checkpoint(
                    args.load_model_path, model)

        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

        model = model.to(device)

        # Train
        train(model, dataloaders, args, device, args.save_dir, log, tbx)
        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

    # Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    dev_results = evaluate(model,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])

    # Log to console
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                 for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))




#train和eval都在
def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """

    # Define loss function

    loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    # Data loaders
    train_loader, domin_labels = dataloaders['train']  # 获取训练数据加载器和domin_labels
    dev_loader = dataloaders['dev']  # 获取验证数据加载器

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer_identity = optim.Adam(params=model.identity.parameters(),lr=args.lr_init, weight_decay=args.l2_wd)
    optimizer = optim.Adam(params=list(model.encoder.parameters()) + list(model.fc.parameters()),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), \
                tqdm(total=total_samples) as progress_bar:
            for x, y, seq_lengths, supports, _, _, patient_id in train_loader:
                batch_size = x.shape[0]
                # patient_id -> domin_label
                batch_domin_labels = torch.tensor(
                    [domin_labels[pid.item()] for pid in patient_id], device=device
                )

                # input sequences
                x = x.to(device)
                y = y.view(-1).to(device)  # (batch_size,)
                #print("y:", y)
                #exit(0)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)


                torch.autograd.set_detect_anomaly(True)
                # Zero out optimizer first
                optimizer_identity.zero_grad()
 

                # Stage 1: Identity classification loss
                _, patient_logits = model(x, seq_lengths, supports)
                
                if patient_logits.shape[-1] == 1:
                    logits = logits.view(-1)  # only classification, pass
                loss_identity = loss_fn(patient_logits, batch_domin_labels)
                 # Backward and update only identity classifier params
                loss_identity.backward(retain_graph=True)
                optimizer_identity.step()

                # Stage 2: Seizure classification loss and adversarial loss
                optimizer.zero_grad()
                logits, _ = model(x, seq_lengths, supports)
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)  # (batch_size,) 
                loss_seizure = loss_fn(logits, y)
                loss = loss_seizure - 0.001 * loss_identity.detach() # 0.001 is the weight for identity loss 
                loss_val = loss.item()


                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += batch_size

               # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    # To evaluate mode
    model.eval()

    # Define loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, seq_lengths, supports, _, file_name, _ in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            #print("y:", y)
            y = y.view(-1).to(device)
            #print("after y:",y)# (batch_size,)
            #exit(0)
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            # (batch_size, num_classes)
            logits, _ = model(x, seq_lengths, supports)

            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )

            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy() # (batch_size, num_classes)
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)

            # Log info
            progress_bar.update(batch_size)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    # print("\n y_pred_all:", y_pred_all.shape,y_pred_all)#103，      5503,
    # print("\n y_true_all:", y_true_all.shape,y_true_all)#103，      5503,
    # print("\n")
    # print("y_prob_all:", y_prob_all.shape, y_prob_all)#103，4     5503,4
    # print("len(y_true_all):", len(set(y_true_all)))
    # exit(0)
    # Threshold search, for detection only
    if (args.task == "detection") and (eval_set == 'dev') and is_test:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        # update dev set y_pred based on best_thresh
        y_pred_all = (y_prob_all > best_thresh).astype(int)  # (batch_size, )
    else:
        best_thresh = best_thresh


    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if args.task == "detection" else "weighted")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh)]
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_args())
 






