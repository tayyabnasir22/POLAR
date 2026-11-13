from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from Trainers.SyntheticTrainer import SyntheticTrainer
from Trainers.RealWorldTrainer import RealWorldTrainer

class TrainingOrchestrator:
    BATCH = 3 # Batch 1 for 16 scale
    REPEAT = 4 # Repeat 8 for 16 scale
    SCALE = 4
    MODEL = ModelType.POLAR
    BENCHMARK = BenchmarkType.NYUV2

    @staticmethod
    def Train():
        print('Batch, Repeat, Scale, Model, Benchmark: ', TrainingOrchestrator.BATCH, TrainingOrchestrator.REPEAT, TrainingOrchestrator.SCALE, TrainingOrchestrator.MODEL, TrainingOrchestrator.BENCHMARK)
    
        if TrainingOrchestrator.BENCHMARK in [BenchmarkType.TOFDSRDReal, BenchmarkType.RGBDDReal]:
            RealWorldTrainer(
                model=TrainingOrchestrator.MODEL,
                benchmark_type=TrainingOrchestrator.BENCHMARK,
                input_patch=448,
                scale=TrainingOrchestrator.SCALE,
                repeat=TrainingOrchestrator.REPEAT,
                batch_size=TrainingOrchestrator.BATCH,
            ).TrainModel()
        else:
            SyntheticTrainer(
                model=TrainingOrchestrator.MODEL,
                benchmark_type=TrainingOrchestrator.BENCHMARK,
                input_patch=448,
                scale=TrainingOrchestrator.SCALE,
                repeat=TrainingOrchestrator.REPEAT, # Change repeat for smaller and larger scales
                batch_size=TrainingOrchestrator.BATCH,
            ).TrainModel()
    