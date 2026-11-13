from Components.POLAR import POLAR
from Components.POLAR_Real import POLAR_Real

from Models.ModelType import ModelType
from Pipelines.PipelineBase import PipelineBase

class BaseModelFactory:
    def __init__(self):
        self._models = {
            ModelType.POLAR: POLAR,
            ModelType.POLAR_Real: POLAR_Real,
        }

    def BuildModel(self, pipeline: PipelineBase, model: ModelType, **kwargs):
        # Create the ecnoder and decoder
        print('Arguments: ', kwargs)
        model = self._models[model](**kwargs).cuda()

        # Call the Pipeline chains
        pipeline.LoadConfigurations()
        pipeline.InitModel(model)
        pipeline.CreateDataLoaders()
        pipeline.LoadModelWeights()
        pipeline.InitTrainingRecipe()
        pipeline.InitModelObjectives()