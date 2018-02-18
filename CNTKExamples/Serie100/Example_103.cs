// MIT Licence (2018)
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
using CNTK;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Example_103
    {
        private Random _random;

        public Example_103()
        {
            _random = new Random(0);
        }

        public void Run()
        {
            var device = DeviceDescriptor.UseDefaultDevice();

            var util = new Example_103_Util();

            // data
            string trainImagesPath = "./Example_103/train-images-idx3-ubyte.gz";
            string trainLabelsPath = "./Example_103/train-labels-idx1-ubyte.gz";
            List<byte[]> trainImages = util.LoadImages(trainImagesPath);
            List<int> trainLabels = util.LoadLabels(trainLabelsPath);
            List<int[]> trainLabels1Hot = trainLabels.Select(x => util.ConvertTo1Hot(x)).ToList();

            string evelImagesPath = "./Example_103/t10k-images-idx3-ubyte.gz";
            string evalLabelsPath = "./Example_103/t10k-labels-idx1-ubyte.gz";
            List<byte[]> evalImages = util.LoadImages(evelImagesPath);
            List<int> evalLabels = util.LoadLabels(evalLabelsPath);
            List<int[]> evalLabels1Hot = evalLabels.Select(x => util.ConvertTo1Hot(x)).ToList();

            // model 

            int sampleSize = trainImages.Count;
            int nbDimensionsInput = trainImages[0].Length;
            int nbLabels = 10; // de 0 à 10

            Variable inputVariables = Variable.InputVariable(NDShape.CreateNDShape(new[] { nbDimensionsInput }), DataType.Double, "input");
            Variable expectedOutput = Variable.InputVariable(NDShape.CreateNDShape(new int[] { nbLabels }), DataType.Double, "output");

            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<double>(1.0/255.0, device), inputVariables);

            Function lastLayer = DefineModel_103C(util, nbLabels, scaledInput);

            Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(lastLayer, expectedOutput);
            Function evalErrorFunction = CNTKLib.ClassificationError(lastLayer, expectedOutput);

            // training

            Trainer trainer;
            {
                // define training

                uint minibatchSize = 64;
                double learningRate = 0.2;
                TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(learningRate, minibatchSize);
                List<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(lastLayer.Parameters(), learningRatePerSample) };
                trainer = Trainer.CreateTrainer(lastLayer, lossFunction, evalErrorFunction, parameterLearners);

                // run training

                int nbSamplesToUseForTraining = trainImages.Count;
                int numSweepsToTrainWith = 10; // traduction de sweep ?
                int numMinibatchesToTrain = nbSamplesToUseForTraining * numSweepsToTrainWith / (int)minibatchSize;

                var minibatchSource = new Example_103_MinibatchSource(inputVariables, trainImages, expectedOutput, trainLabels1Hot, nbSamplesToUseForTraining, numSweepsToTrainWith, minibatchSize, device);
                for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
                {
                    IDictionary<Variable, MinibatchData> data = minibatchSource.GetNextRandomMinibatch();
                    trainer.TrainMinibatch(data, device);
                    util.PrintTrainingProgress(trainer, minibatchCount);
                }
            }

            // evaluate
            {
                uint testMinibatchSize = 512;
                int nbSamplesToTest = evalImages.Count;
                int numMinibatchesToTrain = nbSamplesToTest / (int)testMinibatchSize;
                double testResult = 0;

                var minibatchSource = new Example_103_MinibatchSource(inputVariables, evalImages, expectedOutput, evalLabels1Hot, nbSamplesToTest, 1, testMinibatchSize, device);
                for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
                {
                    IDictionary<Variable, MinibatchData> data = minibatchSource.GetNextRandomMinibatch();

                    UnorderedMapVariableMinibatchData evalInput = new UnorderedMapVariableMinibatchData();
                    foreach (var row in data)
                        evalInput[row.Key] = row.Value;

                    double error = trainer.TestMinibatch(evalInput, device);
                    testResult += error;

                    //var z = CNTKLib.Softmax(lastLayer);
                    //var tOut = new Dictionary<Variable, Value>() { { z.Output, null } };

                    //z.Evaluate(
                    //    new Dictionary<Variable, Value>() { { inputVariables, data[inputVariables].data } },
                    //    tOut,
                    //    device
                    //    );

                    //Value outputValue = tOut[z.Output];
                    //IList<IList<double>> actualLabelSoftMax = outputValue.GetDenseData<double>(z.Output);
                    //var actualLabels = actualLabelSoftMax.Select((IList<double> l) => l.IndexOf(l.Max())).ToList();

                    //Value expectedOutputValue = data[expectedOutput].data;
                    //IList<IList<double>> expectedLabelsSoftmax = expectedOutputValue.GetDenseData<double>(z.Output);
                    //var expectedLabels = expectedLabelsSoftmax.Select((IList<double> l) => l.IndexOf(l.Max())).ToList();

                    //for(int i = 0; i < expectedLabels.Count; i++)
                    //{
                    //    if (actualLabels[i] != expectedLabels[i])
                    //    {
                    //        Debug.WriteLine($"{actualLabels[i]} {expectedLabels[i]}");
                    //    }
                    //}

                    //int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                    Debug.WriteLine($"Average test error: {(testResult / (minibatchCount + 1)):p2}");
                }

                Debug.WriteLine($"Average test error: {(testResult / numMinibatchesToTrain):p2}");
            }
        }

        private Function DefineModel_103B(Example_103_Util util, int nbLabels, Variable inputVariables)
        {
            return util.DenseNode(inputVariables, nbLabels, null);
        }
        
        private Function DefineModel_103C(Example_103_Util util, int nbLabels, Variable inputVariables)
        {
            Func<Variable, Function> activation = CNTKLib.ReLU;
            
            var layer1 = util.DenseNode(inputVariables, 400, activation);
            var layer2 = util.DenseNode(layer1, 200, activation);
            var lastLayer = util.DenseNode(layer2, nbLabels, null); // linear layer
            return lastLayer;
        }
    }
}
