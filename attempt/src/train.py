import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torchvision
import yaml
from tinyimagenet import TinyImageNet
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/nn/')
from simclr import NTXentLoss
from contrastive import SingleInstanceTwoView
from timm_models import build_timm_model

sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
from path_utils import update_config_dirs
from seed import seed_everything
from scheduler import LinearWarmupCosineAnnealingLR
from extend import ExtendedDataset


def print_state_dict(state_dict: dict) -> str:
    state_str = ''
    for key in state_dict.keys():
        if '_loss' in key:
            try:
                state_str += '%s: %.6f. ' % (key, state_dict[key])
            except:
                state_str += '%s: %s. ' % (key, state_dict[key])
        else:
            try:
                state_str += '%s: %.3f. ' % (key, state_dict[key])
            except:
                state_str += '%s: %s. ' % (key, state_dict[key])
    return state_str


def get_dataloaders(
    config: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:
    if config.dataset == 'mnist':
        imsize = 28
        config.in_channels = 1
        config.num_classes = 10
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset = torchvision.datasets.MNIST

    elif config.dataset == 'cifar10':
        imsize = 32
        config.in_channels = 3
        config.num_classes = 10
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10

    elif config.dataset == 'stl10':
        imsize = 96
        config.in_channels = 3
        config.num_classes = 10
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10

    elif config.dataset == 'tinyimagenet':
        imsize = 64
        config.in_channels = 3
        config.num_classes = 200
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = TinyImageNet

    elif config.dataset == 'imagenet':
        imsize = 224
        config.in_channels = 3
        config.num_classes = 1000
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = torchvision.datasets.ImageNet

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            config.dataset)

    transform_train = SingleInstanceTwoView(imsize=imsize,
                                            mean=dataset_mean,
                                            std=dataset_std)

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            imsize,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(imsize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if config.dataset in ['mnist', 'cifar10', 'cifar100']:
        train_dataset = torchvision_dataset(config.dataset_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(config.dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_val)

    elif config.dataset in ['stanfordcars', 'stl10', 'food101', 'flowers102']:
        train_dataset = torchvision_dataset(config.dataset_dir,
                                            split='train',
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(config.dataset_dir,
                                          split='test',
                                          download=True,
                                          transform=transform_val)

        if config.dataset == 'stl10':
            # Training set has too few images (5000 images in total).
            # Let's augment it into a bigger dataset.
            train_dataset = ExtendedDataset(train_dataset,
                                            desired_len=10 *
                                            len(train_dataset))

    elif config.dataset in ['tinyimagenet', 'imagenet']:
        train_dataset = torchvision_dataset(config.dataset_dir,
                                            split='train',
                                            transform=transform_train)
        val_dataset = torchvision_dataset(config.dataset_dir,
                                          split='val',
                                          transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

    return (train_loader, val_loader), config


def train(config: AttributeHashmap) -> None:
    '''
    Train our simple model and record the checkpoints along the training process.
    '''
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)
    train_loader, val_loader = dataloaders

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = '%s/%s-EquiDim%s-TotalDim%s-%s-seed%s.log' % (
        config.log_dir, config.dataset, config.equivariance_dim, config.z_dim,
        config.model, config.random_seed)

    # Log the config.
    config_str = 'Config: \n'
    for key in config.keys():
        config_str += '%s: %s\n' % (key, config[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=log_path, to_console=False)

    model = build_timm_model(model_name=config.model,
                             num_classes=config.num_classes,
                             z_dim=config.z_dim).to(device)
    model.init_params()

    loss_fn_classification = torch.nn.CrossEntropyLoss()
    loss_fn_simclr = NTXentLoss()
    val_metric = 'val_acc'

    opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                            list(model.projection_head.parameters()),
                            lr=float(config.learning_rate))

    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=opt,
                                                 warmup_epochs=min(
                                                     10,
                                                     config.max_epoch // 5),
                                                 max_epochs=config.max_epoch)

    best_val_metric = 0
    best_model = None

    results_dict = {
        'epoch': [],
        'val_acc': [],
    }

    val_metric_pct_list = [20, 30, 40, 50, 60, 70, 80, 90]
    is_model_saved = {}
    for val_metric_pct in val_metric_pct_list:
        is_model_saved[str(val_metric_pct)] = False

    for epoch_idx in tqdm(range(1, config.max_epoch)):
        # For SimCLR, only perform validation / linear probing every 5 epochs.
        skip_epoch_simlr = epoch_idx % 5 != 0

        state_dict = {
            'train_loss': 0,
            'train_acc': 0,
            'val_loss': 0,
            'val_acc': 0,
        }

        state_dict['train_simclr_pseudoAcc'] = 0

        #
        '''
        Training
        '''
        model.train()
        # Because of linear warmup, first step has zero LR. Hence step once before training.
        lr_scheduler.step()
        total_count_loss = 0
        for _, (x, y_true) in enumerate(tqdm(train_loader)):
            # Using SimCLR.
            x_aug1, x_aug2 = x

            B = x_aug1.shape[0]
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x_aug1 = x_aug1.repeat(1, 3, 1, 1)
                x_aug2 = x_aug2.repeat(1, 3, 1, 1)
            x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug2.to(
                device), y_true.to(device)

            # Train encoder.
            z1 = model.project(x_aug1)
            z2 = model.project(x_aug2)

            invariance_dim = z1.shape[1] - config.equivariance_dim
            assert config.equivariance_dim >= 0 and invariance_dim >= 0, \
                'equivariance dim and invariance dim shall sum to Z dim and shall be non-negative.'

            ### Isolation of `invariance` and `equivariance`.

            # `invariance` is the regular contrastive loss.
            if invariance_dim > 0:
                z1_invariance = z1[:, :invariance_dim]
                z2_invariance = z2[:, :invariance_dim]
                loss_invarance, pseudo_acc = loss_fn_simclr(
                    z1_invariance, z2_invariance)
            else:
                loss_invarance = 0
                pseudo_acc = np.nan

            if config.equivariance_dim > 0:
                # The difference in each pair of consecutive data samples shall be similar.
                diff = z1[:, invariance_dim:] - z2[:, invariance_dim:]
                equivariance_vec1 = diff[::2, :]
                equivariance_vec2 = diff[1::2, :]

                loss_equivarance, _ = loss_fn_simclr(equivariance_vec1,
                                                     equivariance_vec2)
            else:
                loss_equivarance = 0

            loss = loss_invarance + loss_equivarance

            state_dict['train_loss'] += loss.item() * B
            state_dict['train_simclr_pseudoAcc'] += pseudo_acc * B
            total_count_loss += B

            opt.zero_grad()
            loss.backward()
            opt.step()

        state_dict['train_simclr_pseudoAcc'] /= total_count_loss
        state_dict['train_loss'] /= total_count_loss

        #
        '''
        Validation (or Linear Probing + Validation)
        '''
        if not skip_epoch_simlr:
            # This function call includes validation.
            probing_acc, val_acc_final = linear_probing(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                device=device,
                loss_fn_classification=loss_fn_classification)
            state_dict['train_acc'] = probing_acc
            state_dict['val_loss'] = np.nan
            state_dict['val_acc'] = val_acc_final
            results_dict['epoch'] = epoch_idx
            results_dict['val_acc'] = val_acc_final
        else:
            state_dict['train_acc'] = 'Val skipped'
            state_dict['val_loss'] = 'Val skipped'
            state_dict['val_acc'] = 'Val skipped'

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        # Save best model
        if not skip_epoch_simlr:
            if state_dict[val_metric] > best_val_metric:
                best_val_metric = state_dict[val_metric]
                best_model = model.state_dict()
                model_save_path = '%s/%s-EquiDim%s-TotalDim%s-%s-seed%s-%s' % (
                    config.checkpoint_dir, config.dataset,
                    config.equivariance_dim, config.z_dim, config.model,
                    config.random_seed, '%s_best.pth' % val_metric)
                torch.save(best_model, model_save_path)
                log('Best model (so far) successfully saved.',
                    filepath=log_path,
                    to_console=False)

                # Save model at each percentile.
                for val_metric_pct in val_metric_pct_list:
                    if state_dict[val_metric] > val_metric_pct and \
                    not is_model_saved[str(val_metric_pct)]:
                        model_save_path = '%s/%s-EquiDim%s-TotalDim%s-%s-seed%s-%s' % (
                            config.checkpoint_dir, config.dataset,
                            config.equivariance_dim, config.z_dim,
                            config.model, config.random_seed, '%s_%s%%.pth' %
                            (val_metric, val_metric_pct))
                        torch.save(best_model, model_save_path)
                        is_model_saved[str(val_metric_pct)] = True
                        log('%s:%s%% model successfully saved.' %
                            (val_metric, val_metric_pct),
                            filepath=log_path,
                            to_console=False)

    # Save the results after training.
    save_path_numpy = '%s/%s-EquiDim%s-TotalDim%s-%s-seed%s/%s' % (
        config.checkpoint_dir, config.dataset, config.equivariance_dim,
        config.z_dim, config.model, config.random_seed, 'results.npz')
    os.makedirs(os.path.dirname(save_path_numpy), exist_ok=True)

    with open(save_path_numpy, 'wb+') as f:
        np.savez(
            f,
            epoch=np.array(results_dict['epoch']),
            val_acc=np.array(results_dict['val_acc']),
        )
    return


def validate_epoch(config: AttributeHashmap,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device):

    correct, total_count_acc = 0, 0
    val_loss, val_acc = 0, 0

    model.eval()
    with torch.no_grad():
        for x, y_true in tqdm(val_loader):
            B = x.shape[0]
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            correct += torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()
            total_count_acc += B

    val_loss = torch.nan
    val_acc = correct / total_count_acc * 100

    return val_loss, val_acc


def linear_probing(config: AttributeHashmap,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module):

    # Separately train linear classifier.
    model.init_linear()
    # Note: Need to create another optimizer because the model will keep updating
    # even after freezing with `requires_grad = False` when `opt` has `momentum`.
    opt_probing = torch.optim.AdamW(list(model.linear.parameters()),
                                    lr=float(config.learning_rate_probing))

    lr_scheduler_probing = LinearWarmupCosineAnnealingLR(
        optimizer=opt_probing,
        warmup_epochs=min(10, config.probing_epoch // 5),
        max_epochs=config.probing_epoch)

    for _ in tqdm(range(config.probing_epoch)):
        # Because of linear warmup, first step has zero LR. Hence step once before training.
        lr_scheduler_probing.step()
        probing_acc = linear_probing_epoch(
            config=config,
            train_loader=train_loader,
            model=model,
            device=device,
            opt_probing=opt_probing,
            loss_fn_classification=loss_fn_classification)

    _, val_acc = validate_epoch(config=config,
                                val_loader=val_loader,
                                model=model,
                                device=device)

    return probing_acc, val_acc


def linear_probing_epoch(config: AttributeHashmap,
                         train_loader: torch.utils.data.DataLoader,
                         model: torch.nn.Module, device: torch.device,
                         opt_probing: torch.optim.Optimizer,
                         loss_fn_classification: torch.nn.Module):
    model.train()
    correct, total_count_acc = 0, 0
    for _, (x, y_true) in enumerate(train_loader):
        x_aug1, x_aug2 = x
        B = x_aug1.shape[0]
        assert config.in_channels in [1, 3]
        if config.in_channels == 1:
            # Repeat the channel dimension: 1 channel -> 3 channels.
            x_aug1 = x_aug1.repeat(1, 3, 1, 1)
            x_aug2 = x_aug2.repeat(1, 3, 1, 1)
        x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug2.to(
            device), y_true.to(device)

        with torch.no_grad():
            h1, h2 = model.encode(x_aug1), model.encode(x_aug2)
        y_pred_aug1, y_pred_aug2 = model.linear(h1), model.linear(h2)
        loss_aug1 = loss_fn_classification(y_pred_aug1, y_true)
        loss_aug2 = loss_fn_classification(y_pred_aug2, y_true)
        loss = (loss_aug1 + loss_aug2) / 2
        correct += torch.sum(
            torch.argmax(y_pred_aug1, dim=-1) == y_true).item()
        correct += torch.sum(
            torch.argmax(y_pred_aug2, dim=-1) == y_true).item()
        total_count_acc += 2 * B

        opt_probing.zero_grad()
        loss.backward()
        opt_probing.step()

    probing_acc = correct / total_count_acc * 100

    return probing_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument(
        '--model',
        help='model name: [resnet, resnext, mobilenet, vit, swin, mobilevit]',
        type=str,
        required=True)
    parser.add_argument('--z-dim',
                        help='total #dimensions in the representation vector',
                        type=int,
                        default=128)
    parser.add_argument(
        '--equivariance-dim',
        help=
        '#dimensions in the representation vector assigned for modeling equivariance',
        type=int,
        default=0)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    parser.add_argument(
        '--random-seed',
        help='Random Seed. If not None, will overwrite config.random_seed.',
        type=int,
        default=None)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.model = args.model
    config.z_dim = args.z_dim
    config.equivariance_dim = args.equivariance_dim
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    config = update_config_dirs(AttributeHashmap(config))

    # Update checkpoint dir.
    config.checkpoint_dir = '%s/%s-%s-%s-seed%s/' % (
        config.checkpoint_dir, config.dataset, config.equivariance_dim,
        config.model, config.random_seed)

    seed_everything(config.random_seed)

    train(config=config)
