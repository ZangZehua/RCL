"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse

import tensorboard_logger as tb_logger
from tensorboardX.writer import SummaryWriter

from torchvision import transforms, datasets
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import ImageFolderInstance

try:
    from apex import amp, optimizers
except ImportError:
    pass

from collections import deque

from algos.ppo_agent import PPO
from config import TrainCMCConfig, train_time


TRAIN_TIME = train_time
LOG_PATH = "saved/CMC_PPO/logs/log-" + TRAIN_TIME + ".txt"
LOG_FILE = open(LOG_PATH, "a+")


def legal_action(action):
    return (0 <= action[0] <= 1.79) \
           and (0 <= action[1] <= 1.79) \
           and (0 <= action[2] <= 1.79) \
           and (0 <= action[3] <= 1.79) \
           and (action[0] <= action[2]) \
           and (action[1] <= action[3]) \
           and (action[2] * action[3] <= 1.0035)


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder, 'unlabeled')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    # PPO agent
    ppo_agent = PPO()

    return model, contrast, criterion_ab, criterion_l, ppo_agent


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, opt, ppo_agent, reward_writer):
    """
    one epoch training
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, indexes) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            indexes = indexes.cuda()
            inputs = inputs.cuda()
        model.eval()
        contrast.eval()
        # ===================rl learning=================
        ppo_agent.set_ac_mode(False)
        ppo_stime = datetime.datetime.now()
        memory_rc = deque()
        memory_hf = deque()
        aug_inputs = torch.tensor([]).cuda()
        all_rewards = []
        illegal_a_num = 0

        for i in range(bsz):
            image = inputs[i].unsqueeze(0)
            index = indexes[i]

            # step 0
            feat_l, feat_ab = model(image)
            state = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)
            out_l, out_ab = contrast(feat_l, feat_ab, index)
            loss_o = criterion_l(out_l.detach()).item() + criterion_ab(out_ab.detach()).item()
            action = ppo_agent.select_action_rc(state)
            legal = legal_action(action)

            if legal:
                action_rc = np.round(np.clip(action, 0.0, 1.79) * 100, 0)
                action_rc[2] = 224 - action_rc[2]
                action_rc[3] = 224 - action_rc[3]
                if i == 0:
                    print(action, action_rc, end=', ')

                image_rc = transforms.functional.resized_crop(image,
                                                              int(action_rc[0]), int(action_rc[1]),
                                                              int(action_rc[2]), int(action_rc[3]),
                                                              [224, 224])
            else:
                illegal_a_num += 1
                if i == 0:
                    print('illegal', end=', ')
                image_rc = transforms.RandomResizedCrop(224, scale=(opt.crop_low, 1.))(image)

            # step 1
            feat_l, feat_ab = model(image_rc)
            state_rc = torch.cat((feat_l.detach(), feat_ab.detach()), 1)
            out_l, out_ab = contrast(feat_l, feat_ab, index)
            loss_rc = criterion_l(out_l.detach()).item() + criterion_ab(out_ab.detach()).item()
            if legal:
                reward = float(loss_o - loss_rc) / 2
            else:
                reward = opt.penalty_reward
            all_rewards.append(reward)
            memory_rc.append([state.cpu().detach().numpy(), action, reward, 1])
            action = ppo_agent.select_action_hf(state_rc)
            if i == 0:
                print(reward, end=', ')
                print(action, end=', ')
            image_hf = transforms.RandomHorizontalFlip(action)(image_rc)

            # step 2
            feat_l, feat_ab = model(image_hf)
            out_l, out_ab = contrast(feat_l.detach(), feat_ab.detach(), index)
            loss_hf = criterion_l(out_l.detach()).item() + criterion_ab(out_ab.detach()).item()
            reward = float(loss_rc - loss_hf)
            all_rewards.append(reward)
            memory_hf.append([state_rc.cpu().detach().numpy(), action, reward, 0])
            if i == 0:
                print(reward, end=',')

            aug_inputs = torch.cat((aug_inputs, image_hf), dim=0)

        ppo_etime = datetime.datetime.now()
        ppo_agent.set_ac_mode(True)
        ppo_agent.learn(memory_rc, memory_hf)
        print("{}, epoch avg reward {}, train time: {}".format(illegal_a_num, np.average(all_rewards), ppo_etime-ppo_stime))
        LOG_FILE.write("{}, epoch avg reward {}, train time: {}\n".format(illegal_a_num, np.average(all_rewards), ppo_etime-ppo_stime))
        reward_writer.add_scalar("train_reward", np.average(all_rewards), (epoch - 1) * len(aug_inputs) + idx)
        # ===================forward=====================
        model.train()
        contrast.train()
        feat_l, feat_ab = model(aug_inputs)
        out_l, out_ab = contrast(feat_l, feat_ab, indexes)

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss
        # print(loss.item(), end=', ')
        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, lprobs=l_prob_meter,
                   abprobs=ab_prob_meter))
            LOG_FILE.write('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, lprobs=l_prob_meter,
                   abprobs=ab_prob_meter))
            LOG_FILE.flush()
            # print(out_l.shape)
            sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():

    # parse the args
    # args = parse_option()
    args = TrainCMCConfig()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l, ppo_agent = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    reward_writer = SummaryWriter(args.reward_writer_path)
    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        LOG_FILE.write("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args, ppo_agent, reward_writer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        LOG_FILE.write('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('l_loss', l_loss, epoch)
        logger.log_value('l_prob', l_prob, epoch)
        logger.log_value('ab_loss', ab_loss, epoch)
        logger.log_value('ab_prob', ab_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            LOG_FILE.write('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
            ppo_agent.save_model(epoch)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
