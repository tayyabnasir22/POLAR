from Models.BenchmarkType import BenchmarkType
from Models.ModelType import ModelType
from ValidationHelpers.SyntheticValidationHelper import SyntheticValidationHelper
from ValidationHelpers.SyntheticValidationHelperBenchmark import SyntheticValidationHelperBenchmark
from ValidationHelpers.RealWorldValidationHelper import RealWorldValidationHelper
from Validators.BaseValidator import BaseValidator

class TestingOrchestrator:
    @staticmethod
    def Validator32x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(32), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelper(32), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_Scale_32', 'last.pth', 449, 32)
        print('-'*33)

    @staticmethod
    def Validator16x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(16), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelper(16), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_Scale_16', 'last.pth', 449, 16)
        print('-'*33)

    @staticmethod
    def Validator8x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(8), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelper(8), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_Scale_8', 'last.pth', 449, 8)
        print('-'*33)

    @staticmethod
    def Validator4x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(4), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelper(4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_4', 'last.pth', 449, 4)
        print('-'*33)

    @staticmethod
    def ValidatorBenchamrk32x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(32), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelperBenchmark(32), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_Scale_32', 'last.pth', 449, 32)
        print('-'*33)
    
    @staticmethod
    def ValidatorBenchamrk16x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(16), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelperBenchmark(16), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_Scale_16', 'last.pth', 449, 16)
        print('-'*33)

    @staticmethod
    def ValidatorBenchamrk8x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(8), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelperBenchmark(8), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_Scale_8', 'last.pth', 449, 8)
        print('-'*33)

    @staticmethod
    def ValidatorBenchamrk4x(data = BenchmarkType):
        print('Running ' + data.value + ', Scale: ', str(4), 'x')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=SyntheticValidationHelperBenchmark(4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/model_states_POLAR_NYUV2_Patch_448_4', 'last.pth', 449, 4)
        print('-'*33)

    @staticmethod
    def ValidatorReal(data = BenchmarkType):
        print('Running ' + data.value + ', Real')
        validator = BaseValidator(
            model=ModelType.POLAR, 
            validation_helper=RealWorldValidationHelper(4), 
            benchmark_type=data
        )
        validator.TestModel('./' + data.value, './StableWeights/NoConfmodel_states_POLAR_Real_RGBDDReal_Patch_448_Scale_1', 'last.pth', 449, 4)
        print('-'*33)
