from DataProcessors.RealWorldProcessor import RealWorldProcessor
from Models.BenchmarkType import BenchmarkType
from Pipelines.Training.BaseTrainingPipeline import BaseTrainingPipeline
from Utilities.DataLoaders import DataLoaders
import os

class RealDataTrainingPipeline(BaseTrainingPipeline):
    def CreateDataLoaders(self,):
        # TOFDSR: loss and RMSE are computed on valid GT pixels only (0.1 m - 5 m)
        base_folder = self.configurations.data_configurations.base_folder
        use_mask = os.path.basename(os.path.normpath(base_folder)) == BenchmarkType.TOFDSRDReal.name
        self.training_data_loader = DataLoaders.GetTrainingDataLoader(
            RealWorldProcessor(
                rgb_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_images_stand_split.npy'),
                depth_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_clipped_split.npy'),
                depth_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_norm_split.npy'),
                mask_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_mask_split.npy'),
                min_max_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_minmax_split.npy'),
                depth_lr_norm_path=os.path.join(self.configurations.data_configurations.base_folder, 'train_depths_lr_norm_split.npy'),
                repeat=self.configurations.data_configurations.repeat,
                augment=self.configurations.data_configurations.augment,
                train=True,
                use_mask=use_mask,
            ),
            self.configurations.data_configurations.batch_size,
        )