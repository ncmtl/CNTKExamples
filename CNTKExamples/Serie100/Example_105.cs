// MIT Licence (2018)
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
using CNTK;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Example_105
    {
        private Random _random;

        public Example_105()
        {
            _random = new Random(0);
        }

        public void Run()
        {
            var device = DeviceDescriptor.UseDefaultDevice();

            var util = new Example_103_Util();

            // data
            string trainImagesPath = "./Example_103/train-images-idx3-ubyte.gz";
            //string trainLabelsPath = "./Example_103/train-labels-idx1-ubyte.gz";
            List<double[]> trainImages = util.LoadImages(trainImagesPath).Select(x => x.Select(y => (double)y).ToArray()).ToList();
            //List<int> trainLabels = util.LoadLabels(trainLabelsPath);
            //List<double[]> trainLabels1Hot = trainLabels.Select(x => util.ConvertTo1Hot(x)).Select(x => x.Cast<double>().ToArray()).ToList();

            string evelImagesPath = "./Example_103/t10k-images-idx3-ubyte.gz";
            //string evalLabelsPath = "./Example_103/t10k-labels-idx1-ubyte.gz";
            List<double[]> evalImages = util.LoadImages(evelImagesPath).Select(x => x.Select(y => (double)y).ToArray()).ToList();
            //List<int> evalLabels = util.LoadLabels(evalLabelsPath);
            //List<int[]> evalLabels1Hot = evalLabels.Select(x => util.ConvertTo1Hot(x)).ToList();

            // model 

            int sampleSize = trainImages.Count;
            int nbDimensionsInput = 28*28;

            Variable inputVariables = Variable.InputVariable(NDShape.CreateNDShape(new [] { nbDimensionsInput }), DataType.Double, "input");
            Variable expectedOutput = Variable.InputVariable(NDShape.CreateNDShape(new [] { nbDimensionsInput }), DataType.Double, "output");

            Function encodeDecode = DefineModel_104B(util, inputVariables, device);

            //var scaledModelOutput = CNTKLib.ElementTimes(Constant.Scalar<double>(1.0 / 255.0, device), encodeDecode);
            //var scaledExpectedOutput = CNTKLib.ElementTimes(Constant.Scalar<double>(1.0 / 255.0, device), expectedOutput);

            //{

            //    Function test = CNTKLib.ElementTimes(
            //                Constant.Scalar(-1.0d, device),
            //                inputVariables);


            //}


            //Function lossFunction = -scaledExpectedOutput * CNTKLib.Log(scaledModelOutput) - (Constant.Scalar(-1.0d, device) - scaledExpectedOutput) * CNTKLib.Log(1 - scaledModelOutput);

            var scaledExpectedOutput = CNTKLib.ElementTimes(expectedOutput, Constant.Scalar(1 / 255.0, device));
            //Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(encodeDecode, expectedOutput);

            // Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(scaledModelOutput, scaledExpectedOutput);
            Function lossFunction = CNTKLib.Square(CNTKLib.Minus(scaledExpectedOutput, encodeDecode));
            
            Function evalErrorFunction = CNTKLib.ClassificationError(encodeDecode, scaledExpectedOutput);

            // training
            
            Trainer trainer;
            {
                // define training
                //int epochSize = 30000;
                uint minibatchSize = 64;
                //double learningRate = 0.8;
                int numSweepsToTrainWith = 2; // traduction de sweep ?
                int nbSamplesToUseForTraining = 60000; // trainImages.Count;

                double lr_per_sample = 0.2;
                //double lr_per_sample = 0.2; // 0.00003;
                //double lr_per_sample = 0.00003; // 0.00003;
                uint epoch_size = 30000; //        # 30000 samples is half the dataset size 

                TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(lr_per_sample, epoch_size);
                TrainingParameterScheduleDouble momentumSchedule = new TrainingParameterScheduleDouble(0.9126265014311797, minibatchSize);

                var parameters = new ParameterVector();
                foreach (var p in encodeDecode.Parameters())
                    parameters.Add(p);
                
                List<Learner> parameterLearners = new List<Learner>() { CNTKLib.FSAdaGradLearner(parameters, learningRatePerSample, momentumSchedule, true) };
                //IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(encodeDecode.Parameters(), learningRatePerSample) };
                trainer = Trainer.CreateTrainer(encodeDecode, lossFunction, evalErrorFunction, parameterLearners);

                // run training

                int numMinibatchesToTrain = nbSamplesToUseForTraining * numSweepsToTrainWith / (int)minibatchSize;

                var minibatchSource = new GenericMinibatchSource(inputVariables, trainImages, expectedOutput, trainImages, nbSamplesToUseForTraining, numSweepsToTrainWith, minibatchSize, device);

                double aggregate_metric = 0;
                for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
                {
                    IDictionary<Variable, MinibatchData> data = minibatchSource.GetNextRandomMinibatch();
                    trainer.TrainMinibatch(data, device);

                    double samples = trainer.PreviousMinibatchSampleCount();
                    double avg = trainer.PreviousMinibatchEvaluationAverage();
                    aggregate_metric += avg * samples;
                    double nbSampleSeen = trainer.TotalNumberOfSamplesSeen();
                    double train_error = aggregate_metric / nbSampleSeen;
                    Debug.WriteLine($"{minibatchCount} Average training error: {train_error:p2}");
                }
            }

            // evaluate
            {
                uint testMinibatchSize = 32;
                int nbSamplesToTest = 32;// evalImages.Count;
                int numMinibatchesToTrain = nbSamplesToTest / (int)testMinibatchSize;

                double metric_numer = 0;
                double metric_denom = 0;

                var minibatchSource = new GenericMinibatchSource(inputVariables, evalImages, expectedOutput, evalImages, nbSamplesToTest, 1, testMinibatchSize, device);
                for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
                {
                    IDictionary<Variable, MinibatchData> data = minibatchSource.GetNextRandomMinibatch();

                    ////UnorderedMapVariableMinibatchData evalInput = new UnorderedMapVariableMinibatchData();
                    ////foreach (var row in data)
                    ////    evalInput[row.Key] = row.Value;

                    ////double error = trainer.TestMinibatch(evalInput, device);
                                        
                    ////metric_numer += Math.Abs(error * testMinibatchSize);
                    ////metric_denom += testMinibatchSize;

                    ////MinibatchData outputValue = evalInput[expectedOutput];

                    //IList<IList<double>> inputPixels = outputValue.data.GetDenseData<double>(inputVariables);

                    //IList<IList<double>> actualLabelSoftMax = outputValue.data.GetDenseData<double>(encodeDecode.Output);

                    //for (int i = 0; i < actualLabelSoftMax.Count; i++)
                    //    PrintBitmap(inputPixels[i], actualLabelSoftMax[i], i);

                    // var normalizedInput = CNTKLib.ElementTimes(Constant.Scalar<double>(1.0 / 255.0, device), inputVariables);

                    Dictionary<Variable, Value> input = new Dictionary<Variable, Value>()
                    {
                        { inputVariables, data[inputVariables].data }
                    };
                    Dictionary<Variable, Value> output = new Dictionary<Variable, Value>()
                    {
                        // { normalizedInput.Output, null }
                        { encodeDecode.Output, null }
                    };
                                        
                    encodeDecode.Evaluate(input, output, device);
                    
                    IList<IList<double>> inputPixels = input[inputVariables].GetDenseData<double>(inputVariables);
                    IList<IList<double>> outputPixels = output[encodeDecode.Output].GetDenseData<double>(encodeDecode.Output);
                    for (int i = 0; i < inputPixels.Count; i++)
                        PrintBitmap(inputPixels[i], outputPixels[i], i);


                }

                double test_error = (metric_numer * 100.0) / (metric_denom);
                Debug.WriteLine($"Average test error: {test_error:p2}");
            }
        }

        private void PrintBitmap(IList<double> source, IList<double> output, int index)
        {
            Bitmap bitmap = new Bitmap(28*2, 28);
            for(int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                {
                    int pixelIndex = i + 28 * j;
                    int sourcePixel = (int)(source[pixelIndex]);
                    bitmap.SetPixel(i, j, Color.FromArgb(sourcePixel, sourcePixel, sourcePixel));

                    int outputPixel = ((int)(output[pixelIndex] * 255));
                    bitmap.SetPixel(i+28, j, Color.FromArgb(outputPixel, outputPixel, outputPixel));
                }
            string filename = @"C:\Users\Nelson\Desktop\tmp\ttt\ttt_{0}.bmp".Replace("{0}", index.ToString());
            if (File.Exists(filename))
                File.Delete(filename);
            bitmap.Save(filename);
        }

        private Function DefineModel_104A(Example_103_Util util, Variable inputVariables, DeviceDescriptor device)
        {
            var normalizedInput = CNTKLib.ElementTimes(Constant.Scalar<double>(1.0 / 255.0, device), inputVariables);
            var encode3 = util.DenseNode(normalizedInput, 32, CNTKLib.ReLU);                        
            var decoder = util.DenseNode(encode3, 28 * 28, CNTKLib.Sigmoid);

            return decoder;
        }


        private Function DefineModel_104B(Example_103_Util util, Variable inputVariables, DeviceDescriptor device)
        {
            var normalizedInput = CNTKLib.ElementTimes(Constant.Scalar<double>(1.0 / 255.0, device), inputVariables);
            var encode1 = util.DenseNode(normalizedInput, 128, CNTKLib.ReLU);
            var encode2 = util.DenseNode(encode1, 64, CNTKLib.ReLU);
            var encode3 = util.DenseNode(encode2, 32, CNTKLib.ReLU);


            var decode3 = util.DenseNode(encode3, 32, CNTKLib.ReLU);
            var decode2 = util.DenseNode(decode3, 64, CNTKLib.ReLU);
            var decode1 = util.DenseNode(decode2, 128, CNTKLib.ReLU);
            var decoder = util.DenseNode(decode1, 28 * 28, CNTKLib.Sigmoid);

            return (decoder);
        }
    }
}
