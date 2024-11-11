import numpy as np
import os
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader, TensorDataset
from args import get_args
from model.graph_sleep_net.MSTGCN import MSTGCN
from model.graph_sleep_net.Utils import cheb_polynomial, scaled_Laplacian
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import KMeans
from collections import Counter

# 定义训练操作
def main(args):
    # Get device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=123)

    # Get save directories
    args.save_dir = utils.get_save_dir(args.save_dir, training=True if args.do_train else False)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # Load feature data from .npz file
    log.info('Loading feature data...')
    if args.task == 'detection':
        data = np.load("Feature/Feature_detection.npz")
    elif args.task == 'classification':
        data = np.load("Feature/Feature_classification.npz")
    else:
        raise ValueError('Task not supported: {}'.format(args.task))
    train_features = data['train_feature']
    val_features = data['val_feature']
    test_features = data['test_features']
    train_labels = data['train_targets']
    val_labels = data['val_targets']
    test_labels = data['test_targets']
    train_domin = data['train_domin']
    val_domin = data['val_domin']
    test_domin = data['test_domin']

    train_domin = train_domin.tolist()
    val_domin = val_domin.tolist()
    test_domin = test_domin.tolist()

    # Ensure patient IDs are hashable by converting to tuples
    train_domin = [tuple(id) if isinstance(id, list) else id for id in train_domin]
    val_domin = [tuple(id) if isinstance(id, list) else id for id in val_domin]
    test_domin = [tuple(id) if isinstance(id, list) else id for id in test_domin]
    ########################################################################################
    # Combine all domin data to perform clustering
    all_domin = train_domin + val_domin + test_domin

    # Count the frequency of each patient ID
    counter = Counter(all_domin)
    freq_threshold = 8  # Set frequency threshold
    low_freq_patient_ids = [patient_id for patient_id, freq in counter.items() if freq < freq_threshold]

    # Perform KMeans clustering on low frequency patient IDs
    low_freq_patient_ids_indices = [i for i, patient_id in enumerate(all_domin) if patient_id in low_freq_patient_ids]
    low_freq_patient_ids_data = np.array([all_domin[i] for i in low_freq_patient_ids_indices]).reshape(-1, 1)

    # Execute KMeans clustering
    n_clusters = 5  # Number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(low_freq_patient_ids_data)
    clusters = kmeans.labels_

    # Replace original low frequency patient IDs with cluster labels
    for i, index in enumerate(low_freq_patient_ids_indices):
        all_domin[index] = f'cluster_{clusters[i]}'

    # Split the modified domin data back into train, val, and test
    train_domin = all_domin[:len(train_domin)]
    val_domin = all_domin[len(train_domin):len(train_domin) + len(val_domin)]
    test_domin = all_domin[-len(test_domin):]
    ########################################################################################
    # Map patient IDs to continuous indices
    unique_train_patient_ids = list(set(train_domin))
    id_to_index = {id: idx for idx, id in enumerate(unique_train_patient_ids)}

    train_domin_indexed = np.array([id_to_index[id] for id in train_domin])
    val_domin_indexed = np.array([id_to_index.get(id, -1) for id in val_domin])  # 如果ID不存在，则设为-1
    test_domin_indexed = np.array([id_to_index.get(id, -1) for id in test_domin])  # 如果ID不存在，则设为-1






    # Create data loaders
    log.info('Building dataset...')
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32),
                                  torch.tensor(train_labels, dtype=torch.long),
                                  torch.tensor(train_domin_indexed, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32),
                                torch.tensor(val_labels, dtype=torch.long),
                                torch.tensor(val_domin_indexed, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.long),
                                 torch.tensor(test_domin_indexed, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    dataloaders = {'train': train_loader, 'dev': val_loader, 'test': test_loader}


    adj_data = np.load(os.path.join('data/electrode_graph/', 'adj_matrices.npz'))
    adj_matrix = adj_data['adj_matrix']
    cheb_polynomials = [torch.tensor(p, dtype=torch.float32).to(device) for p in cheb_polynomial(scaled_Laplacian(adj_matrix), 3)]

    # Build model
    log.info('Building model...')
    num_vertices = train_features.shape[1]  # 19
    num_features = train_features.shape[2]  # 640
    num_domains = len(unique_train_patient_ids)  # 确保 num_domains 是正确的


    #print("num_of_train_patient_ids:", len(train_patient_ids))
    #print("num_of_unique_patient_ids:", len(unique_patient_ids))
    #print("train_domin_index_len", len(train_domin_indexed))

    model = MSTGCN(num_vertices, num_features, 1, 1, 64, 64,
                   3, 1, 3, 0.0001, cheb_polynomials, args.num_classes, num_domains).to(device)

    num_params = utils.count_parameters(model)
    log.info('Total number of trainable parameters: {}'.format(num_params))

    if args.do_train:
        if args.load_model_path is not None:
            model = utils.load_model_checkpoint(args.load_model_path, model)

        # Train
        train(model, dataloaders, args, device, args.save_dir, log, tbx)
        # Load best model after training finished
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

    # Evaluate on dev and test set
    log.info('Training DONE. Evaluating model...')
    dev_results = evaluate(model, dataloaders['dev'], args, args.save_dir, device, is_test=True, nll_meter=None,
                           eval_set='dev')
    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v) for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model, dataloaders['test'], args, args.save_dir, device, is_test=True, nll_meter=None,
                            eval_set='test', best_thresh=dev_results['best_thresh'])
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v) for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))


def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """
    # Define loss function

    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
        #DominLoss = nn.CrossEntropyLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)
        #DominLoss = nn.CrossEntropyLoss().to(device)

    # Dominance loss
    DominLoss = nn.CrossEntropyLoss().to(device)
    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    # Get saver
    saver = utils.CheckpointSaver(save_dir, metric_name=args.metric_name, maximize_metric=args.maximize_metric, log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Dictionary to keep track of domain label frequencies
    domain_label_count = {}

    # Train
    log.info('Training...')
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch != 80) and (not early_stop): #args.num_epochs
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), tqdm(total=total_samples) as progress_bar:
            for x, y, domin in train_loader:
                batch_size = x.shape[0]

                # for label in domin.cpu().numpy():
                #     if label in domain_label_count:
                #         domain_label_count[label] += 1
                #     else:
                #         domain_label_count[label] = 1


                # input seqs
                x = x.to(device)
                #print("test",x.size, x.type(), y.size, y.type(),y[0])
                y = y.view(-1).to(device).long() if args.num_classes > 1 else y.to(device).float()
                domin = domin.to(device).long()
                # (batch_size, )

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                logits,domin_out = model(x)
                #print("Output logits:", logits)
                #print("Output logits size:", logits.size())
                loss_domin = DominLoss(domin_out, domin)

                loss = loss_fn(logits, y) + 0.01 * loss_domin
                loss_val = loss.item()


                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val, lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))
                eval_results = evaluate(model, dev_loader, args, save_dir, device, is_test=False, nll_meter=nll_meter)
                best_path = saver.save(epoch, model, optimizer, eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Early stop
                if (patience_count == args.patience):
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v) for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()
        # log.info('Domain label frequencies:')
        # for label, count in domain_label_count.items():
        #     log.info('Label {}: {} times'.format(label, count))


def evaluate(model, dataloader, args, save_dir, device, is_test=False, nll_meter=None, eval_set='dev', best_thresh=0.5):
    # To evaluate mode
    model.eval()

    # Define loss function
    if args.task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, domin in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.view(-1).to(device).long() if args.num_classes > 1 else y.view(-1).to(device).float()
            #y = y.to(device).float()

            # Forward
            logits, domin_out = model(x)


            if args.num_classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

                # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

            # Log info
            progress_bar.update(batch_size)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    y_true_all_list = y_true_all


    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all_list,
                                        y_prob=y_prob_all,
                                        file_names=None,
                                        average="binary" if args.task == "detection" else "weighted")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss), ('acc', scores_dict['acc']), ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']), ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh)]
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results


if __name__ == '__main__':
    main(get_args())

