// MIT Licence (2018)
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
using CNTK;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Example_201
    {
        public void Run()
        { 
            var device = DeviceDescriptor.UseDefaultDevice();
            var util = new Example_103_Util();

            Example_201_Data datasource = new Example_201_Data();
            IEnumerable<Example_201_Item> trainingImages = datasource.LoadTrainingImages().ToList();
            IEnumerable<Example_201_Item> testImages = datasource.LoadTestImages().ToList();
            IDictionary<double, string> labelIndex = datasource.LoadLabelIndex().ToDictionary(x => (double) x.Key, x => x.Value);

            int image_height = 32, image_width = 32, num_channels = 3, num_classes = 10;

            Variable input = Variable.InputVariable(NDShape.CreateNDShape(new[] { image_height, image_width, num_channels }), DataType.Double, "input");
            Variable expectedOutput = Variable.InputVariable(new int[] { num_classes }, DataType.Double, "expectedOutput");

            Function normalizedInput = CNTKLib.ElementTimes(Constant.Scalar(1.0 / 255.0, device), input);
            Function model = DefineModel_C(normalizedInput, num_classes, util);

            Variable output = model.Output;

            uint minibatchSize = 64;
            Trainer trainer = MakeTrainer(expectedOutput, output, model, minibatchSize);
            
            {   // train
                int nbSamplesToUseForTraining = trainingImages.Count();
                int numSweepsToTrainWith = 5;
                int numMinibatchesToTrain = nbSamplesToUseForTraining * numSweepsToTrainWith / (int)minibatchSize;
                var trainingInput = trainingImages.Select(x => x.Image.Select(y => (double)y).ToArray()).ToList();
                var trainingOutput = trainingImages.Select(x => ToOneHotVector(x.Label, labelIndex.Count)).ToList();
                var trainingMinibatchSource = new GenericMinibatchSource(input, trainingInput, expectedOutput, trainingOutput, nbSamplesToUseForTraining, numSweepsToTrainWith, minibatchSize, device);
                RunTraining(trainer, trainingMinibatchSource, numMinibatchesToTrain, device);
            }

            // evaluate
            Evaluate(model, testImages, input, device, labelIndex);
        }

        private double[] ToOneHotVector(byte label, int nbLabels)
        {
            double[] vector = new double[nbLabels];
            vector[label] = 1;
            return vector;
        }

        private void Evaluate(Function model, IEnumerable<Example_201_Item> evalData2, Variable input, DeviceDescriptor device, IDictionary<double, string> labelIndex)
        {
            List<Example_201_Item> orderedData = evalData2.ToList();

            int minibatchSize = 32;

            List<Example_201_Item> queue = new List<Example_201_Item>();
            List<int> predictedLabels = new List<int>();

            for (int i = 0; i < orderedData.Count; i++)
            {
                queue.Add(orderedData[i]);
                if(queue.Count == minibatchSize || i == orderedData.Count -1)
                {
                    List<int> tempPred = EvaluateBatch(model, queue, input, device);
                    predictedLabels.AddRange(tempPred);
                    queue.Clear();
                }
            }
                        
            if (predictedLabels.Count != orderedData.Count)
                throw new Exception("actualLabels.Count != orderedData.Count");

            Dictionary<int, int> nbSuccessByLabel = new Dictionary<int, int>();
            Dictionary<int, int> nbItemsByLabel = new Dictionary<int, int>();

            int nbSuccess = 0;
            for (int i = 0; i < predictedLabels.Count; i++)
            {
                int predictedClassification = predictedLabels[i];
                int expectedClassification = (int)orderedData[i].Label;

                if (!nbItemsByLabel.ContainsKey(expectedClassification))
                    nbItemsByLabel[expectedClassification] = 0;

                nbItemsByLabel[expectedClassification]++;

                if (predictedClassification == expectedClassification)
                {
                    nbSuccess++;
                    if (!nbSuccessByLabel.ContainsKey(expectedClassification))
                        nbSuccessByLabel[expectedClassification] = 0;
                    nbSuccessByLabel[expectedClassification]++;
                }                               
            }

            Debug.WriteLine($"Prediction accuracy : {nbSuccess / (double)orderedData.Count:p2}");

            for(int i = 0; i < labelIndex.Count; i++)
            {
                int nbSuccessByLabelI;
                if (!nbSuccessByLabel.TryGetValue(i, out nbSuccessByLabelI))
                    nbSuccessByLabelI = 0;

                Debug.WriteLine($"Prediction accuracy (label = {labelIndex[i]}): { nbSuccessByLabelI / (double)nbItemsByLabel[i]:p2}");

            }

        }

        private List<int> EvaluateBatch(Function model, IEnumerable<Example_201_Item> evalData2, Variable input, DeviceDescriptor device)
        {
            List<Example_201_Item> orderedData = evalData2.ToList();
            Value inputData = Value.CreateBatch(input.Shape, orderedData.SelectMany(data => data.Image.Select(x => (double)x)).ToArray(), device);
            var inputDic = new Dictionary<Variable, Value>() { { input, inputData } };
            var outputDic = new Dictionary<Variable, Value>() { { model.Output, null } };

            model.Evaluate(inputDic, outputDic, device);

            Value outputValue = outputDic[model.Output];

            IList<IList<double>> prediction = outputValue.GetDenseData<double>(model.Output);
            List<int> predictedLabels = prediction.Select((IList<double> l) => l.IndexOf(l.Max())).ToList();

            return predictedLabels;
        }
                
        private void RunTraining(Trainer trainer, GenericMinibatchSource minibatchSource, int numMinibatchesToTrain, DeviceDescriptor device)
        {
            Debug.WriteLine($"Minibatch;CrossEntropyLoss;EvaluationCriterion;");
            double aggregate_metric = 0;
            for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
            {
                IDictionary<Variable, MinibatchData> data = minibatchSource.GetNextRandomMinibatch();
                trainer.TrainMinibatch(data, device);
                PrintTrainingProgress(trainer, minibatchCount);
            }
        }
        
        public void PrintTrainingProgress(Trainer trainer, int minibatchIdx)
        {
            if (trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                Debug.WriteLine($"{minibatchIdx};{trainLossValue};{evaluationValue};");
            }
        }

        private Function DefineModel_A(Variable input, int nbLabels, Example_103_Util util)
        {
            Func<Variable, Function> activation = CNTKLib.ReLU;

            var layer1 = util.Convolution(input, new[] { 5, 5 }, 32, activation);
            var layer2 = util.PoolingMax(layer1, new[] { 3, 3 }, new[] { 2, 2 });

            var layer3 = util.Convolution(layer2, new[] { 5, 5 }, 32, activation);
            var layer4 = util.PoolingMax(layer3, new[] { 3, 3 }, new[] { 2, 2 });

            var layer5 = util.Convolution(layer4, new[] { 5, 5 }, 64, activation);
            var layer6 = util.PoolingMax(layer5, new[] { 3, 3 }, new[] { 2, 2 });

            var layer7 = util.DenseNode(layer6, 64, activation); // linear layer
            var lastLayer = util.DenseNode(layer7, nbLabels, null); // linear layer

            return lastLayer;            
        }


        private Function DefineModel_B(Variable input, int nbLabels, Example_103_Util util)
        {
            Func<Variable, Function> activation = CNTKLib.ReLU;

            var layer1 = util.Convolution(input, new[] { 5, 5 }, 32, activation);
            var layer2 = util.PoolingMax(layer1, new[] { 3, 3 }, new[] { 2, 2 });

            var layer3 = util.Convolution(layer2, new[] { 5, 5 }, 32, activation);
            var layer4 = util.PoolingMax(layer3, new[] { 3, 3 }, new[] { 2, 2 });

            var layer5 = util.Convolution(layer4, new[] { 5, 5 }, 64, activation);
            var layer6 = util.PoolingMax(layer5, new[] { 3, 3 }, new[] { 2, 2 });

            var layer7 = util.DenseNode(layer6, 64, activation); // linear layer
            var dropout = CNTKLib.Dropout(layer7, 0.25);
            var lastLayer = util.DenseNode(dropout, nbLabels, null); // linear layer

            return lastLayer;
        }

        private Function DefineModel_C(Variable input, int nbLabels, Example_103_Util util)
        {
            Func<Variable, Function> activation = CNTKLib.ReLU;

            var layer1 = util.Convolution(input, new[] { 5, 5 }, 32, activation);
            var bn1 = util.BatchNormalization(layer1);
            var layer2 = util.PoolingMax(bn1, new[] { 3, 3 }, new[] { 2, 2 });

            var layer3 = util.Convolution(layer2, new[] { 5, 5 }, 32, activation);
            var bn3 = util.BatchNormalization(layer3);
            var layer4 = util.PoolingMax(bn3, new[] { 3, 3 }, new[] { 2, 2 });

            var layer5 = util.Convolution(layer4, new[] { 5, 5 }, 64, activation);
            var bn5 = util.BatchNormalization(layer5);
            var layer6 = util.PoolingMax(bn5, new[] { 3, 3 }, new[] { 2, 2 });

            var layer7 = util.DenseNode(layer6, 64, activation); // linear layer
            var bn7 = util.BatchNormalization(layer7);
            var lastLayer = util.DenseNode(bn7, nbLabels, null); // linear layer

            return lastLayer;
        }

        private Function DefineModel_D_Pourri(Variable input, int nbLabels, Example_103_Util util)
        {
            Func<Variable, Function> activation = CNTKLib.ReLU;

            Variable net = input;

            foreach (int dims in new[] { 64 , 96, 128 })
            {
                net = util.Convolution(net, new[] { 3, 3 }, dims, activation);
                net = util.Convolution(net, new[] { 3, 3 }, dims, activation);
                net = util.PoolingMax(net, new[] { 3, 3 }, new int[] { 2, 2 });
            }

            for (int i = 0; i < 2; i++)
                net = util.DenseNode(net, 1024, activation);

            net = util.DenseNode(net, nbLabels, CNTKLib.Softmax);
            
            return net;            
        }

        private Function DefineModel_E(Variable input, int nbLabels, Example_103_Util util)
        {
            Func<Variable, Function> activation = CNTKLib.ReLU;
            
            Variable net = input;

            net = util.ConvolutionWithBatchNormalization(net, new[] { 3, 3 }, 16, activation);
            net = util.ResNetBasicStack(net, 16, 3, activation);

            net = util.ResNetInc(net, 32, activation);
            net = util.ResNetBasicStack(net, 32, 2, activation);

            net = util.ResNetInc(net, 64, activation);
            net = util.ResNetBasicStack(net, 64, 2, activation);

            net = util.PoolingMax(net, new[] { 8, 8 }, new[] { 1, 1 });
            net = util.DenseNode(net, nbLabels, null);

            return net;
        }

        private Trainer MakeTrainer(Variable expectedOutput, Variable output, Function model, uint minibatchSize)
        {
            double learningRate = 0.01;
            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(learningRate);
            TrainingParameterScheduleDouble momentumSchedule = new TrainingParameterScheduleDouble(0.9983550962823424, minibatchSize);

            //Function lossFunction = CrossEntropyWithSoftmax(output, expectedOutput);
            Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(output, expectedOutput);
            Function evalErrorFunction = CNTKLib.ClassificationError(output, expectedOutput);
            
            var parameters = new ParameterVector();
            foreach (var p in model.Parameters())
                parameters.Add(p);

            List<Learner> parameterLearners = new List<Learner>() { CNTKLib.MomentumSGDLearner(parameters, learningRatePerSample, momentumSchedule, true, new AdditionalLearningOptions() {l2RegularizationWeight=0.001 }) };
            Trainer trainer = Trainer.CreateTrainer(model, lossFunction, evalErrorFunction, parameterLearners);

            return trainer;
        }

        //Function CrossEntropyWithSoftmax(Variable output, Variable label)
        //{
        //    var p = CNTKLib.Softmax(output);
        //    //var axisVector = new AxisVector(output.Shape.Dimensions.Select(ax => new Axis(ax)).ToArray());
        //    var cews = CNTKLib.Minus(CNTKLib.ReduceLogSum(p, new Axis(0)), CNTKLib.TransposeTimes(label, p));
        //    return cews;
        //}
    }
}
