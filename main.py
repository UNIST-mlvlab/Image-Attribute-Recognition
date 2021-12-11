import argparse
import time
import sys
import shutil

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision

import model as models
from data_utils.datasets import Get_Dataset
from data_utils.description import *
from utils.utils import *



def get_args():
    parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
    parser.add_argument('--data_path', default='./data_path', type=str)
    parser.add_argument('--label_data_path', default='./data_utils', type=str)
    parser.add_argument('--experiment', default='rap', type=str)
    parser.add_argument('--approach', default='inception_iccv', type=str)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--eps', default=1e-12, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--decay_epoch', default=(20,40), type=eval)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualization attention localization module')

    return parser.parse_args()


def main():
    global args
    args = get_args()
    shutil.rmtree('./runs')
    writer = SummaryWriter()

    # Seed
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    else:
        print('[CUDA unavailable]'); sys.exit()

    print('=' * 100)
    print('Arguments = ')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    if args.vis:
        # Data loading code
        train_dataset, val_dataset = Get_Dataset(args.experiment, args.data_path, args.label_data_path)
        val_loader = DataLoader(val_dataset,
                                batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # create model
        model = torch.load('model/saved_model.pt')
        model.eval()

        # get the number of model parameters
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        print()

        for i, (input, target) in enumerate(val_loader):
            if i % 10000 == 0:
                raw_img = torchvision.utils.make_grid(input[0], normalize=True)
                writer.add_image(f'img_{i}/_raw_image', raw_img, 0)
                model.module.visualization(input, str(i), writer, target.squeeze())
            #else:
                #model.module.visualization(input, str(i))

        writer.close()

    else:
        # Data loading code
        train_dataset, val_dataset = Get_Dataset(args.experiment, args.data_path, args.label_data_path)
        attr_num = data_info[args.experiment]['attr_nums']
        description = data_info[args.experiment]['attr_list']

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        val_loader = DataLoader(val_dataset,
                                batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # create model
        model = models.__dict__[args.approach](pretrained=True, num_classes=attr_num)

        # get the number of model parameters
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        print()

        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        model = torch.nn.DataParallel(model).cuda()

        best_accu = 0
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_accu = checkpoint['best_accu']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = False
        cudnn.deterministic = True

        # define loss function
        criterion = Weighted_BCELoss(args.experiment, data_info['rap']['weights'], args.eps)

        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        if args.evaluate:
            test(val_loader, model, attr_num, description)

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(args.lr, optimizer, epoch, args.decay_epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            accu = validate(val_loader, model, criterion, epoch)

            test(val_loader, model, attr_num, description)

            # remember best Accu and save checkpoint
            is_best = accu > best_accu
            best_accu = max(accu, best_accu)

            if epoch in args.decay_epoch:
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accu': best_accu,
                }, epoch+1, args.prefix)

            torch.save(model, 'trained_model.pt')


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        output = model(input)

        bs = target.size(0)

        if type(output) == type(()) or type(output) == type([]):
            loss_list = []
            # deep supervision
            for k in range(len(output)):
                out = output[k]
                loss_list.append(criterion.forward(torch.sigmoid(out), target, epoch))
            loss = sum(loss_list)
            # maximum voting
            output = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])
        else:
            loss = criterion.forward(torch.sigmoid(output), target, epoch)

        # measure accuracy and record loss
        accu = accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()

    end = time.time()
    for i, _ in enumerate(val_loader):
        input, target = _
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        output = model(input)

        bs = target.size(0)

        if type(output) == type(()) or type(output) == type([]):
            loss_list = []
            # deep supervision
            for k in range(len(output)):
                out = output[k]
                loss_list.append(criterion.forward(torch.sigmoid(out), target, epoch))
            loss = sum(loss_list)
            # maximum voting
            output = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])
        else:
            loss = criterion.forward(torch.sigmoid(output), target, epoch)

        # measure accuracy and record loss
        accu = accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Accu {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def test(val_loader, model, attr_num, description):
    model.eval()

    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)

    for i, _ in enumerate(val_loader):
        input, target = _
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        output = model(input)
        bs = target.size(0)

        # maximum voting
        if type(output) == type(()) or type(output) == type([]):
            output = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])


        batch_size = target.size(0)
        tol = tol + batch_size
        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        target = target.cpu().numpy()

        for it in range(attr_num):
            for jt in range(batch_size):
                if target[jt][it] == 1:
                    pos_tol[it] = pos_tol[it] + 1
                    if output[jt][it] == 1:
                        pos_cnt[it] = pos_cnt[it] + 1
                if target[jt][it] == 0:
                    neg_tol[it] = neg_tol[it] + 1
                    if output[jt][it] == 0:
                        neg_cnt[it] = neg_cnt[it] + 1

        if attr_num == 1:
            continue
        for jt in range(batch_size):
            tp = 0
            fn = 0
            fp = 0
            for it in range(attr_num):
                if output[jt][it] == 1 and target[jt][it] == 1:
                    tp = tp + 1
                elif output[jt][it] == 0 and target[jt][it] == 1:
                    fn = fn + 1
                elif output[jt][it] == 1 and target[jt][it] == 0:
                    fp = fp + 1
            if tp + fn + fp != 0:
                accu = accu +  1.0 * tp / (tp + fn + fp)
            if tp + fp != 0:
                prec = prec + 1.0 * tp / (tp + fp)
            if tp + fn != 0:
                recall = recall + 1.0 * tp / (tp + fn)

    print('=' * 100)
    print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    mA = 0.0
    for it in range(attr_num):
        cur_mA = ((1.0*pos_cnt[it]/pos_tol[it]) + (1.0*neg_cnt[it]/neg_tol[it])) / 2.0
        mA = mA + cur_mA
        print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
    mA = mA / attr_num
    print('\t' + 'mA:        '+str(mA))

    if attr_num != 1:
        accu = accu / tol
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall)
        print('\t' + 'Accuracy:  '+str(accu))
        print('\t' + 'Precision: '+str(prec))
        print('\t' + 'Recall:    '+str(recall))
        print('\t' + 'F1_Score:  '+str(f1))
    print('=' * 100)



if __name__ == '__main__':
    main()