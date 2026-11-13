from DataProcessors.DataProcessorBase import DataProcessorBase
from torch.utils.data import DataLoader

class DataLoaders:
    @staticmethod
    def GetTrainingDataLoader(data_processor: DataProcessorBase, batch_size: int = 16):
        return DataLoader(
            data_processor, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
            persistent_workers=True
        )

    @staticmethod
    def GetValidationDataLoader(data_processor: DataProcessorBase, batch_size: int = 16):
        return DataLoader(
            data_processor, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )

    @staticmethod
    def GetTestingDataLoader(data_processor: DataProcessorBase, batch_size: int = 16):
        return DataLoader(
            data_processor, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )