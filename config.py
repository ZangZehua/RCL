import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"

import torch
import datetime

train_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H-%M-%S"))


class TrainCMCConfig:
    # base config
    print_freq = 10
    tb_freq = 500
    save_freq = 10
    batch_size = 256
    num_workers = 5
    # batch_size = 32
    # num_workers = 0
    epochs = 240

    # optimization
    learning_rate = 0.03
    lr_decay_epochs = [120, 160, 200]
    lr_decay_rate = 0.1
    beta1 = 0.5
    beta2 = 0.999
    weight_decay = 1e-4
    momentum = 0.9

    # resume path
    resume = ''

    # model definition
    model = 'alexnet'
    softmax = False
    nce_k = 16384
    nce_t = 0.07
    nce_m = 0.5
    feat_dim = 128

    # dataset
    dataset = 'STL-10'  # ??? STL-10

    # specify folder
    # data_folder = "/content/drive/MyDrive/projects/datasets/STL-10"
    data_folder = "/home/zzh/datasets/STL-10"
    # data_folder = "d:/projects/data/STL-10"
    model_path = "saved/CMC_PPO/models"
    tb_path = "runs"

    # add new views
    view = 'Lab'

    # mixed precision setting
    amp = False
    opt_level = '02'

    # data crop threshold
    crop_low = 0.2

    # penalty reward
    penalty_reward = -10.0

    method = 'softmax' if softmax else 'nce'
    model_name = 'memory_{}_{}_{}_lr_{}_decay_{}_bsz_{}'.format(method, nce_k, model, learning_rate,
                                                                weight_decay, batch_size)
    if amp:
        model_name = '{}_amp_{}'.format(model_name, opt_level)
    model_name = '{}_view_{}'.format(model_name, view)
    model_folder = os.path.join(model_path, model_name)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    tb_folder = os.path.join(tb_path, model_name)
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    if not os.path.isdir(data_folder):
        raise ValueError('data path not exist: {}'.format(data_folder))
    # save config
    if not os.path.exists("runs"):
        os.mkdir("runs")
    if not os.path.exists("runs/CMC_PPO"):
        os.mkdir("runs/CMC_PPO")
    reward_writer_path = "runs/CMC_PPO/" + train_time + "_train-reward"
    rl_loss_writer_path = "runs/CMC_PPO/" + train_time + "_rl-loss"
    cl_loss_writer_path = "runs/CMC_PPO/" + train_time + "_cl-loss"

    if not os.path.exists("saved"):
        os.mkdir("saved")
    if not os.path.exists("saved/CMC_PPO"):
        os.mkdir("saved/CMC_PPO")
    if not os.path.exists("saved/CMC_PPO/models"):
        os.mkdir("saved/CMC_PPO/models")
    actor_model_save_path = "saved/CMC_PPO/models/" + train_time + "_actor.pth"
    critic_model_save_path = "saved/CMC_PPO/models/" + train_time + "_critic.pth"
    cmc_model_save_path = "saved/CMC_PPO/models/" + train_time + "_cmc.pth"

    if not os.path.exists("saved/CMC_PPO/logs"):
        os.mkdir("saved/CMC_PPO/logs")
    log_path = "saved/CMC_PPO/logs/log-" + train_time + ".txt"


class EvalCMCConfig:
    print_freq = 10
    tb_freq = 500
    save_freq = 5
    batch_size = 32
    num_workers = 0
    epochs = 60

    learning_rate = 0.1
    lr_decay_epochs = [30, 40, 50]
    lr_decay_rate = 0.2
    momentum = 0.9
    weight_decay = 0
    beta1 = 0.5
    beta2 = 0.999

    resume = ''

    model = 'alexnet'
    model_path = 'saved/CMC_PPO/models/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_Lab/ckpt_epoch_240.pth'

    ppo_model_path_list = [
        "saved/CMC_PPO/models/actor_rc.pth",
        "saved/CMC_PPO/models/critic_rc.pth",
        "saved/CMC_PPO/models/actor_hf.pth",
        "saved/CMC_PPO/models/critic_hf.pth"
    ]
    layer = 5
    dataset = 'STL-10'
    view = 'Lab'
    data_folder = "/home/zzh/datasets/STL-10"
    save_path = "saved/CMC_PPO/models"
    tb_path = "runs"

    crop_low = 0.2
    log = 'saved/CMC_PPO/logs/time_linear.txt'
    gpu = 0

    if (data_folder is None) or (save_path is None) or (tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | save_path | tb_path')

    if dataset == 'imagenet':
        if 'alexnet' not in model:
            crop_low = 0.08

    model_name = model_path.split('/')[-2]
    model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(model_name, batch_size, learning_rate,
                                                              weight_decay)

    model_name = '{}_view_{}'.format(model_name, view)
    tb_folder = os.path.join(tb_path, model_name + '_layer{}'.format(layer))
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    save_folder = os.path.join(save_path, model_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if dataset == 'imagenet100':
        n_label = 100
    if dataset == 'imagenet':
        n_label = 1000
    if dataset == 'STL-10':
        n_label = 10


class RunnerConfig:
    # common config
    random_seed = 27  # set random seed if required (0 = no random seed)
    torch.manual_seed(random_seed)

    # cuda config
    device_ids = [0]

    # train config
    memory_size = 10000

    update_target_interval = 100
    print_interval = update_target_interval
    log_interval = update_target_interval


class PPOConfig:
    actor_lr = 0.0003
    critic_lr = 0.0003
    weight_decay = 0.001
    observation_dim = 128 * 2
    cmc_batch_size = 32
    k_epoch = 10

    # resized crop config
    rc_action_dim = 4

    # horizontal flip config
    hf_action_dim = 2
    batch_size = 64
    gamma = 0.99
    lamda = 0.98
    clip_param = 0.2
    update_interval = 2000
    update_episode = 50


class PPONetworkConfig:  # for ppo_continuous and ppo_discrete
    hidden_dim_1 = 128
    hidden_dim_2 = 64

