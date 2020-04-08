## This code will replace final layers and train new data (transfer learning).

from __future__ import print_function
from __future__ import division
from torchvision.models import densenet121, resnet152



import matplotlib
matplotlib.use('Agg')


import pprint
import torch.optim as optim
import os


import argparse, json
from Dataset_feature_extraction import Dataset_FE as data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm
from tensorboardX import SummaryWriter

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == 'resnet':
        net = resnet152()
        set_parameter_requires_grad(net, feature_extract)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        grad_cam_hooks = {'forward': net.layer4, 'backward': net.fc}

        input_size=224

    elif model_name == 'densenet':
        net = densenet121()
        set_parameter_requires_grad(net, feature_extract)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=num_classes)
        input_size=224

    return net, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return model

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def save_checkpoint(checkpoint, optim_checkpoint, args, max_records=10):
    """ save model and optimizer checkpoint along with csv tracker
    of last `max_records` best number of checkpoints as sorted by avg auc """
    # 1. save latest
    torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
    torch.save(optim_checkpoint, os.path.join(args.output_dir, 'optim_checkpoint_latest.pt'))
    #if sched_checkpoint: torch.save(sched_checkpoint, os.path.join(args.output_dir, 'sched_checkpoint_latest.pt'))

    # 2. save the last `max_records` number of checkpoints as sorted by avg auc
    tracker_path = os.path.join(args.output_dir, 'checkpoints_tracker.csv')
    tracker_header = ' '.join(['CheckpointId', 'Step', 'Loss', 'AvgAUC'])

    # 2a. load checkpoint stats from file
    old_data = None             # init and overwrite from records
    file_id = 0                 # init and overwrite from records
    lowest_auc = float('-inf')  # init and overwrite from records
    if os.path.exists(tracker_path):
        old_data = np.atleast_2d(np.loadtxt(tracker_path, skiprows=1))
        file_id = len(old_data)
        if len(old_data) == max_records: # remove the lowest-roc record and add new checkpoint record under its file-id
            lowest_auc_idx = old_data[:,3].argmin()
            lowest_auc = old_data[lowest_auc_idx, 3]
            file_id = int(old_data[lowest_auc_idx, 0])
            old_data = np.delete(old_data, lowest_auc_idx, 0)

    # 2b. update tracking data and sort by descending avg auc
    data = np.atleast_2d([file_id, args.step, checkpoint['eval_loss'], checkpoint['avg_auc']])
    if old_data is not None: data = np.vstack([old_data, data])
    data = data[data.argsort(0)[:,3][::-1]]  # sort descending by AvgAUC column

    # 2c. save tracker and checkpoint if better than what is already saved
    if checkpoint['avg_auc'] > lowest_auc:
        np.savetxt(tracker_path, data, delimiter=' ', header=tracker_header)
        torch.save(checkpoint, os.path.join(args.output_dir, 'best_checkpoints', 'checkpoint_{}.pt'.format(file_id)))

def train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, epoch, args):
    model.train()

    with tqdm(total=len(train_dataloader), desc='Step at start {}; Training epoch {}/{}'.format(args.step, epoch+1, args.n_epochs)) as pbar:
        for x, target, idxs in train_dataloader:
            args.step += 1
            out = model(x.to(args.device))
            loss = loss_fn(out, target.to(args.device)).sum(1).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss = '{:.4f}'.format(loss.item()))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_loss', loss.item(), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

            # evaluate and save on eval_interval
            if args.step % args.eval_interval == 0:
                with torch.no_grad():
                    model.eval()

                    eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)

                    writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
                    for k, v in eval_metrics['aucs'].items():
                        writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

                    # save model
                    save_checkpoint(checkpoint={'global_step': args.step,
                                                'eval_loss': np.sum(list(eval_metrics['loss'].values())),
                                                'avg_auc': np.nanmean(list(eval_metrics['aucs'].values())),
                                                'state_dict': model.state_dict()},
                                    optim_checkpoint=optimizer.state_dict(),
                                    #sched_checkpoint=scheduler.state_dict() if scheduler else None,
                                    args=args)

                    # switch back to train mode
                    model.train()
def compute_metrics(outputs, targets, losses):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:,i], outputs[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics
@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    targets, outputs, losses = [], [], []
    for x, target, idxs in dataloader:
        out = model(x.to(args.device))
        loss = loss_fn(out, target.to(args.device))

        outputs += [out.cpu()]
        targets += [target]
        losses  += [loss.cpu()]

    return torch.cat(outputs), torch.cat(targets), torch.cat(losses)

def evaluate_single_model(model, dataloader, loss_fn, args):
    outputs, targets, losses = evaluate(model, dataloader, loss_fn, args)
    return compute_metrics(outputs, targets, losses)

def compute_metrics(outputs, targets, losses):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:,i], outputs[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics



def train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, args):
    for epoch in range(args.n_epochs):
        # train
        train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, writer, epoch, args)

        # evaluate
        print('Evaluating...', end='\r')
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics @ step {}:'.format(args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
        for k, v in eval_metrics['aucs'].items():
            writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

        # save eval metrics
        save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)


def main(args):
    writer = SummaryWriter(logdir=args.output_dir)

    # Initialize the basic model for this run.
    model_ft, input_size = initialize_model(args.model_name, args.num_classes_orig, args.feature_extract)
    if args.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            #if name=='features.norm5.wight' or name =='features.norm5.bias' or name == "features.denseblock4.denselayer16.conv2.weight" or name=='features.denseblock4.denselayer16.norm2.bias':
                #param.requires_grad = True   # Make the layers whose weights need to be changed as reuires grad.
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Get the loaded state dictionaries.
    print('Restoring model weights from {}'.format(args.restore))
    model_checkpoint = torch.load(args.restore, map_location=args.device)
    model_ft.load_state_dict(model_checkpoint['state_dict'])

    model_ft.classifier = nn.Linear(model_ft.classifier.in_features, out_features=args.num_classes_final)
    args.step = model_checkpoint['global_step']
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001)

    # if training, load optimizer.
    if args.train:
        print('Restoring optimizer.')
        optim_checkpoint_path = os.path.join(os.path.dirname(args.restore), 'optim_' + os.path.basename(args.restore))
        optimizer_ft.load_state_dict(torch.load(optim_checkpoint_path, map_location=args.device))
    # Creating the dataloader for loading data
    # You need a train loader that should output the targets as attributes as a tensor of float 32's (multiclass allowed) and the image as PIL read image.
    print("Initializing Datasets and Dataloaders...")
    train_dataset = data(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, drop_last=False)
    val_dataset = data(mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False, drop_last=False)
    # Setup the loss fxn
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(args.device)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Send the model to GPU
    model_ft = model_ft.to(device)
    # # Train and evaluate
    model_ft, hist = train_and_evaluate(model_ft, train_loader, val_loader, criterion, optimizer_ft, writer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--model_name', type=str, default='densenet', help='model name for network training')
    parser.add_argument('--data_dir', type=str, default='', help='data containing the pretraining folder')
    parser.add_argument('--output_dir', type=str, default='/home/cougarnet.uh.edu/mpadmana/PycharmProjects/glomerulus_classification/save/openi_feature_extraction_results_4')
    parser.add_argument('--num_classes_orig', type=int, default= 5, help='Number of classes/ units in the output layer')
    parser.add_argument('--num_classes_final', type=int, default= 17, help='Number of classes/ units in the output layer')
    parser.add_argument('--batch_size', type=int, default= 16, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default= 30, help='Number of epochs')
    parser.add_argument('--feature_extract', type=bool, default= True, help='True if you want feature extraction, false if you want finetuning')
    parser.add_argument('--restore', type=str, default='/home/cougarnet.uh.edu/mpadmana/PycharmProjects/glomerulus_classification/save/train_and_val_chexpert_5_classes/checkpoint_latest.pt', help='The restored model. Transfer learned model')
    parser.add_argument('--train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=40)
    parser.add_argument('--eval_interval', type=int, default=20)
    main(parser.parse_args())
