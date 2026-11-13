from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from ValidationHelpers.ValidationHelperBase import ValidationHelperBase
from ModelFactories.BaseModelFactory import BaseModelFactory
from Models.RunningAverage import RunningAverage
from Models.Timer import Timer
from Pipelines.Validation.BaseTestingPipeline import BaseTestingPipeline
from Pipelines.Validation.BaseTestingPipelineBenchmark import BaseTestingPipelineBenchmark
from Pipelines.Validation.RealWorldTestingPipeline import RealWorldTestingPipeline
from Utilities.Logger import Logger
from ValidationHelpers.SyntheticValidationHelper import SyntheticValidationHelper
import os

class BaseValidator:
    def __init__(self, model: ModelType, validation_helper: ValidationHelperBase, benchmark_type: BenchmarkType):
        self._model = model
        self._validation_helper = validation_helper
        self._benchmark_type = benchmark_type

    def _RunTests(
            self, 
            pipeline: BaseTestingPipeline, 
            factory: BaseModelFactory, 
        ):
        factory.BuildModel(pipeline, self._model, use_pretrained = False, num_feats=48, 
            img_size=448, 
            patch_size=16, 
            scale=self._validation_helper.scale,
            train_mode=False
        )

        # 4. Call the testing
        timer = Timer()
        results: dict[str, RunningAverage] = self._GetPrediction(pipeline)

        # 5. Pass the result to writer
        out_path = Logger.LogTestResultsToCSV(
            results, 
            self._benchmark_type, 
            pipeline.configurations.data_configurations.eval_scale, 
            os.path.dirname(pipeline.configurations.model_path), 
            timer, 
            pipeline.configurations.model_path, 
        )

        Logger.Log('Test results saved at: ', out_path)

    def TestModel(self, valid_data_path, model_load_path, model_name, total_example, eval_scale):
        # 1. Init thre required Pipeline
        print(valid_data_path, model_load_path)
        if self._benchmark_type in [BenchmarkType.LU, BenchmarkType.MIDDLE]:
            pipeline = BaseTestingPipelineBenchmark(
                    valid_data_path=valid_data_path,
                model_load_path=model_load_path,
                model_name=model_name,
                total_example=total_example,
                eval_scale=eval_scale,
                patch_size_valid=None,
            )
        elif self._benchmark_type in [BenchmarkType.RGBDDReal, BenchmarkType.TOFDSRDReal]:
            pipeline = RealWorldTestingPipeline(
                valid_data_path=valid_data_path,
                model_load_path=model_load_path,
                model_name=model_name,
                total_example=total_example,
                eval_scale=eval_scale,
                patch_size_valid=None,
            )
        else:
            pipeline = BaseTestingPipeline(
                valid_data_path=valid_data_path,
                model_load_path=model_load_path,
                model_name=model_name,
                total_example=total_example,
                eval_scale=eval_scale,
                patch_size_valid=None,
            )

        # 2. Build the model
        factory = BaseModelFactory()
        
        self._RunTests(pipeline, factory)
    
    def _GetPrediction(
            self, 
            pipeline: BaseTestingPipeline, 
        ):
        return self._validation_helper.EvaluteForTesting(
                pipeline.validation_data_loader, 
                pipeline.model
            )
