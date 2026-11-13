from TestingOrchestrators.TestingOrchestrator import TestingOrchestrator
from Models.BenchmarkType import BenchmarkType

def main():
      # TestingOrchestrator.Validator4x(BenchmarkType.DIML)
      # TestingOrchestrator.Validator4x(BenchmarkType.NYUV2)
      # TestingOrchestrator.Validator4x(BenchmarkType.RGBDD)
      # TestingOrchestrator.Validator4x(BenchmarkType.TOFDSRD)
      # TestingOrchestrator.ValidatorBenchamrk4x(BenchmarkType.MIDDLE)
      # TestingOrchestrator.ValidatorBenchamrk4x(BenchmarkType.LU)

      # TestingOrchestrator.Validator8x(BenchmarkType.DIML)
      # TestingOrchestrator.Validator8x(BenchmarkType.NYUV2)
      # TestingOrchestrator.Validator8x(BenchmarkType.RGBDD)
      # TestingOrchestrator.Validator8x(BenchmarkType.TOFDSRD)
      # TestingOrchestrator.ValidatorBenchamrk8x(BenchmarkType.MIDDLE)
      # TestingOrchestrator.ValidatorBenchamrk8x(BenchmarkType.LU)

      # TestingOrchestrator.Validator16x(BenchmarkType.DIML)
      # TestingOrchestrator.Validator16x(BenchmarkType.NYUV2)
      # TestingOrchestrator.Validator16x(BenchmarkType.RGBDD)
      # TestingOrchestrator.Validator16x(BenchmarkType.TOFDSRD)
      # TestingOrchestrator.ValidatorBenchamrk16x(BenchmarkType.MIDDLE)
      # TestingOrchestrator.ValidatorBenchamrk16x(BenchmarkType.LU)

      # TestingOrchestrator.Validator32x(BenchmarkType.DIML)
      # TestingOrchestrator.Validator32x(BenchmarkType.NYUV2)
      # TestingOrchestrator.Validator32x(BenchmarkType.RGBDD)
      # TestingOrchestrator.Validator32x(BenchmarkType.TOFDSRD)
      # TestingOrchestrator.ValidatorBenchamrk32x(BenchmarkType.MIDDLE)
      # TestingOrchestrator.ValidatorBenchamrk32x(BenchmarkType.LU)


      TestingOrchestrator.ValidatorReal(BenchmarkType.RGBDDReal)
      TestingOrchestrator.ValidatorReal(BenchmarkType.TOFDSRDReal)

if __name__ == '__main__':
    main()