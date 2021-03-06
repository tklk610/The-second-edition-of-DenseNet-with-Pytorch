import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        if is_best:
            best_pred = float(state['best_pred'])

            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

            if self.runs:
                previous_acc = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')

                    if os.path.exists(path):
                        with open(path, 'r') as f :
                            acc = float(f.readline())
                            previous_acc.append(acc)
                    else :
                        continue

                max_acc = max(previous_acc)
                if best_pred > max_acc:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))


    def save_experiment_config(self):
        logfile  = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset']               = self.args.dataset
        p['backbone']             = self.args.backbone
        p['img_size']             = self.args.img_size
        p['batch']                = self.args.batch
        p['epochs']               = self.args.epochs
        p['adam']                 = self.args.adam
        p['lr']                   = self.args.lr
        p['ddp']                  = self.args.ddp
        p['dsconv']               = self.args.dsconv
        p['active']               = self.args.active
        p['lr_scheduler']         = self.args.lr_scheduler
        p['loss_type']            = self.args.loss_type
        p['momentum']             = self.args.momentum
        p['weigth_decay']         = self.args.weight_decay
        p['nesterov']             = self.args.nesterov
        #p['use_balanced_weights'] = self.args.use_balanced_weights
        p['gpu_ids']              = self.args.gpu_ids
        p['workers']              = self.args.workers

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
