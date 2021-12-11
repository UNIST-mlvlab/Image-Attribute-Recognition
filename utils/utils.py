import os
import torch
import numpy as np



def save_checkpoint(args, state, epoch, prefix, filename='.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "saved_parameters" + args.experiment + '/' + args.approach + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if prefix == '':
        filename = directory + str(epoch) + filename
    else:
        filename = directory + prefix + '_' + str(epoch) + filename
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, optimizer, epoch, decay_epoch):
    for epc in decay_epoch:
        if epoch >= epc:
            lr = lr * 0.1
        else:
            break
    print()
    print('Learning Rate:', lr)
    print()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        res.append(1.0*sum(correct[:,k]) / batch_size)
    return sum(res) / attr_num


class Weighted_BCELoss(object):
    """
        Weighted_BCELoss was proposed in "Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios"[13].
    """
    def __init__(self, experiment, weights, eps):
        super(Weighted_BCELoss, self).__init__()
        self.eps = eps
        self.weights = None
        if experiment == 'rap':
            self.weights = torch.Tensor(weights).cuda()

    def forward(self, output, target, epoch):
        if self.weights is not None:
            cur_weights = torch.exp(target + (1 - target * 2) * self.weights)
            loss = cur_weights * (target * torch.log(output + self.eps)) + ((1 - target) * torch.log(1 - output + self.eps))
        else:
            loss = target * torch.log(output + self.eps) + (1 - target) * torch.log(1 - output + self.eps)
        return torch.neg(torch.mean(loss))