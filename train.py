import argparse
import os
from tqdm import tqdm

from dataloaders import cfg
import torch.distributed as dist
from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.densenet import *
from utils.loss import SegmentationLosses
#from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer  = self.summary.create_summary()

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DenseNet(
            backbone    = args.backbone,
            compression = args.compression,
            num_classes = cfg.NUM_CLASSES,
            bottleneck  = args.bottleneck,
            drop_rate   = args.drop_rate,
            sync_bn     = args.sync_bn,
            dsconv      = args.dsconv,
            training    = args.training,
            amp_sel     = args.amp_sel,
            active      = args.active
        )

        print(model)

        device = torch.device("cuda:{}".format(args.local_rank))

        # Print number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        print("Total parameters: ", num_params)

        # Define Optimizer
        if args.adam :
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr           = args.lr,
                betas        = (0.9, 0.999),
                eps          = 1e-08,
                weight_decay = args.weight_decay
            )
        else :
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr           = args.lr,
                momentum     = args.momentum,
                weight_decay = args.weight_decay,
                nesterov     = args.nesterov
            )

        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        # self.evaluator = Evaluator(self.nclass)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        if args.cuda and args.ddp :
            # self.model = self.model.cuda()
            self.model = self.model.to(device)
            # 同步BN
            self.model = convert_syncbn_model(model)
            # 混合精度
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level)
            # 分布数据并行
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)

        # Resuming checkpoint
        self.best_pred = 0.0

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("[Note] : no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda and args.ddp :
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.opt_level)
                self.model.load_state_dict(checkpoint['state_dict'])  # 注意，load模型需要在amp.initialize之后！！！
                optimizer.load_state_dict(checkpoint['optimizer'])
                amp.load_state_dict(checkpoint['amp'])

                if args.local_rank  == 0 :
                    print("[Note] : APEX DDP MODE is used!!!")

            elif args.cuda and not args.ddp:
                self.model.module.load_state_dict(checkpoint['state_dict'])
                print("[Note] : Resume CUDA is used!!!")
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.best_pred = checkpoint['best_pred']
            print("[Note] : Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0


    def training(self, epoch) :
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        if self.args.local_rank == 0:
            print("num_img_tr = %d" %(num_img_tr))

        for i, (image, target) in enumerate(tbar) :
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss   = self.criterion(output, target)
            #loss   = F.cross_entropy(output, target)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss :
                scaled_loss.backward()

            self.optimizer.step()
            train_loss += loss.item()

            if self.args.local_rank == 0:
                tbar.set_description('Train Loss: %.5f' % (train_loss / (i + 1)))
                self.writer.add_scalar('Train/Total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #      global_step = i + num_img_tr * epoch
            #      self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        if self.args.local_rank == 0:
            self.writer.add_scalar('Train_loss/Total_epoch', train_loss, epoch)
            print('[Epoch/Total Epochs: %d/%d, numImages: %5d]' % (epoch, self.args.epochs, i * self.args.batch + image.data.shape[0]))
            print('Total Train Loss: %.5f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False

            if self.args.ddp :
                if self.args.local_rank == 0 :
                    self.saver.save_checkpoint({
                        'epoch'      : epoch + 1,
                        'state_dict' : self.model.module.state_dict(),
                        'optimizer'  : self.optimizer.state_dict(),
                        'amp'        : amp.state_dict(),
                        'best_pred'  : self.best_pred,
                    }, is_best)
            else :
                self.saver.save_checkpoint({
                    'epoch'      : epoch + 1,
                    'state_dict' : self.model.module.state_dict(),
                    'optimizer'  : self.optimizer.state_dict(),
                    'best_pred'  : self.best_pred,
                }, is_best)


    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        Acc  = 0.0
        #Acc_class = 0.0
        val_loss = 0.0

        for i, (image, target) in enumerate(tbar) :
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            loss      = self.criterion(output, target)
            val_loss += loss.item()

            if self.args.local_rank == 0:
                tbar.set_description('Val Loss: %.5f' %(val_loss / (i + 1)))
            #pred = output.data.cpu().float()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            Acc += (pred == target).sum().float()/self.args.batch
            # All_acc += len(target)

            # Add batch sample into evaluator
            # self.evaluator.add_batch(target, pred)

        # Fast test during the training

        Acc /= len(tbar)

        if self.args.local_rank == 0:
            self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
            self.writer.add_scalar('val/Acc',              Acc,      epoch)
            #self.writer.add_scalar('val/Acc_class',        Acc_class, epoch)

            print('Validation:')
            print('[Epoch / Total Epochs: %d/%d, numImages: %5d]' % (epoch, self.args.epochs, i * self.args.batch + image.data.shape[0]))
            print("Acc:{}".format(Acc))
            print('Total Val Loss: %.5f' %val_loss)

        new_pred = Acc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if self.args.ddp :
                if self.args.local_rank == 0 :
                    self.saver.save_checkpoint({
                        'epoch'      : epoch + 1,
                        'state_dict' : self.model.module.state_dict(),
                        'optimizer'  : self.optimizer.state_dict(),
                        'amp'        : amp.state_dict(),
                        'best_pred'  : self.best_pred,
                    }, is_best)
            else :
                self.saver.save_checkpoint({
                    'epoch'      : epoch + 1,
                    'state_dict' : self.model.module.state_dict(),
                    'optimizer'  : self.optimizer.state_dict(),
                    'best_pred'  : self.best_pred,
                }, is_best)


def main():
    # 命令行交互，设置一些基本的参数
    parser = argparse.ArgumentParser(description="PyTorch DenseNet Training")
    parser.add_argument('--backbone', type=str, default='net121', choices=['net121', 'net161', 'net169', 'net201'],
                                                                           help='backbone name')
    parser.add_argument('--compression', type=int, default=0.7, help='network output stride')
    parser.add_argument('--bottleneck', type=str, default=True, help='network output stride')
    parser.add_argument('--drop_rate', type=int, default=0.5, help='dropout rate')
    parser.add_argument('--training', type=str, default=True, help='')
    parser.add_argument('--dataset', type=str, default='oled_data', choices=['oled_data'],
                                                                   help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--img_size', type=int, default=(320, 320), help='train and val image resize')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal'],
                                                                          help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                                                                  help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch', type=int, default=None, metavar='N', help='input batch size for \
                                                                                training (default: auto)')
    parser.add_argument('--test_batch', type=int, default=None, metavar='N', help='input batch size for \
                                                                                testing (default: auto)')
    #parser.add_argument('--use_balanced_weights', action='store_true', default=True,
    #                                                    help='whether to use balanced weights (default: False)')

    # optimizer params
    parser.add_argument('--adam', default=False, help='use torch.optim.Adam() optimizer')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                                                                      help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=None, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True, help='whether use nesterov (default: False)')

    # APEX DDP mode
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--ddp', type=str, default=True, help='use APEX DDP mode')
    parser.add_argument('--opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'], help='use AMP mode')
    parser.add_argument('--active', type=str, default='ReLU', choices=['SWISH', 'PReLU', 'ReLU'],
                                                       help='Selection of a activation function')

    # cuda, seed and logging
    parser.add_argument('--sync_bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
    parser.add_argument('--dsconv', type=bool, default=False, help='whether to use deep separable conv ')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='use which gpu to train, must be a \
                                                                     comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=True, help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.amp_sel  = args.opt_level != 'O0'

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
            print("[Note] : Sync Batch is used!!!")
        else:
            args.sync_bn = False

    if args.weight_decay is None :
        if args.active == 'PReLU' :
            args.weight_decay = 0
        else:
            args.weight_decay = 5e-4

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'oled_data'  : 100,
            'cityscapes' : 200,
            'pascal'     : 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch is None:
        args.batch = 4 * len(args.gpu_ids)

    if args.test_batch is None:
        args.test_batch = args.batch

    if args.lr is None:
        lrs = {
            'coco'       : 0.1,
            'cityscapes' : 0.01,
            'pascal'     : 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch

    if args.checkname is None:
        args.checkname = str(args.backbone) + '_' + str(args.batch) + '_' + str(args.epochs)

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
