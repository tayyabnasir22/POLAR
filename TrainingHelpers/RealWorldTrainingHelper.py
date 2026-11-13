from Components.SimpleGradientL1Loss import SimpleGradientL1Loss
from Models.RunningAverage import RunningAverage
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Pipelines.Validation.BaseTestingPipelineBenchmark import BaseTestingPipelineBenchmark
from Utilities.Evaluation import Evaluation
from Utilities.Logger import Logger
from ValidationHelpers.RealWorldValidationHelper import RealWorldValidationHelper
from tensorboardX import SummaryWriter
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from TrainingHelpers.TrainingHelperBase import TrainingHelperBase
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch


class RealWorldTrainingHelper(TrainingHelperBase):
    def __init__(self, pipeline: BaseTrainingPipeline, validation_pipeline: BaseTestingPipeline, benchmark_pipeline: BaseTestingPipelineBenchmark, scale: int  = 4, patch_size: int = 256):
        super().__init__(
            pipeline=pipeline,
            benchmark_pipeline=benchmark_pipeline,
            validation_pipeline=validation_pipeline,
            validation_metric_seed=99999
        )
        self.validation_helper = RealWorldValidationHelper(scale)

        self.benchmark_validation_helper = None

        self.loss = SimpleGradientL1Loss(lambda_grad=0.1)
        
    def RunEpoch(
        self,
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: Optimizer,
        epoch: int, 
        configurations: TrainingDataConfigurations, 
        writer = SummaryWriter
    ):
        # 1. Run the single epoch
        model.train()

        # 2. Calculate steps for the current epoch
        total_steps = int(configurations.total_examples / configurations.batch_size * configurations.repeat)
        step = 0

        # 3. Init the running average
        all_loss_avg = RunningAverage()
        rmse_sr_avg = RunningAverage()

        # 4. Process loss for each batch
        for batch in tqdm(train_loader, leave=False, desc='train'):
            # 4.1. convert all items to cuda
            for k, v in batch.items():
                if k not in ['orig_h', 'orig_w']:
                    batch[k] = v.cuda(non_blocking=True)

            # 4.2 Get output from the model
            out, clean_lr, boundaries = model(batch['rgb'], batch['lr'])

            # 4.3 Get L1 loss b/w GT and predicted HR
            valid = batch['mask'] > 0.5
            valid_lr = batch['mask_lr'] > 0.5
            gt_target = torch.where(valid, batch['gt_norm'], out.detach())
            clean_lr_target = torch.where(valid_lr, batch['clean_lr_norm'], clean_lr.detach())
            
            loss = self.loss(out, gt_target, clean_lr, clean_lr_target, boundaries)
            
            # 4.7 Log the epoch-step, loss, and metric
            evals = {}
            evals['loss'] = loss
            Logger.LogSummaryWriter(writer, 'train', evals, epoch, step, total_steps)
            step += 1

            # 4.8 Accumulate the loss for the step
            all_loss_avg.SetItem(loss.item())

            train_rmse = Evaluation.DepthRMSE(out, batch['gt_norm'], batch['max'], batch['min'], mask=batch['mask'])
            if train_rmse == train_rmse: # skip NaN (batch without any valid GT pixel)
                rmse_sr_avg.SetItem(train_rmse)
            
            # 4.9 Back propogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.isfinite(loss):
                pass
            else:
                if not torch.isfinite(loss):
                    print("NaN detected main loss")

                print("Pred min/max:", out.min().item(), out.max().item())
                print("GT min/max:", batch['gt'].min().item(), batch['gt'].max().item())
                raise RuntimeError("Stopping due to NaN")

        print('train: loss={:.4f}'.format(all_loss_avg.GetItem()))

        print('train: RMSE={:.4f}'.format(rmse_sr_avg.GetItem()))

        # 5. Return the accumulated loss
        return all_loss_avg.GetItem()
    
    def Validation(
        self,
        model, 
        log_info: list, 
        epoch: int, 
        validation_metric_seed: float
    ):
        validation_metrics: dict = self.validation_helper.EvaluateForTrainigData(self._validation_pipeline.validation_data_loader, model)
        
        for key, value in validation_metrics.items():
            log_info.append('Validation ' + key + '= {:.4f}'.format(value))
        
    def BenachmarkValidation(
        self,
        model, 
        log_info: list, 
        epoch: int, 
        validation_metric_seed: float
    ):
        pass