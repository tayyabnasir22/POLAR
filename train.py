from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from TrainingOrchestrators.TrainingOrchestrator import TrainingOrchestrator
import sys
from Utilities.PathManager import PathManager

def main(scale, model):
    models = {
        'polar': ModelType.POLAR,
        'real': ModelType.POLAR_Real
    }


    PathManager.BASE_PATH = './'
    TrainingOrchestrator.SCALE = scale
    TrainingOrchestrator.MODEL = models[model]
    TrainingOrchestrator.BENCHMARK = BenchmarkType.RGBDDReal

    TrainingOrchestrator.REPEAT = 2 # 1 for HYPERSIM x8, 2 for all else
    TrainingOrchestrator.BATCH = 2
    
    TrainingOrchestrator.Train()

if __name__ == '__main__':
    args_len = len(sys.argv) - 1

    main(int(sys.argv[1]), int(sys.argv[2]))