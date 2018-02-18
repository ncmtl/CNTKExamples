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
    class Example_101_B
    {
        class DataPoint
        {
            public double Age { get; set; }
            public double TumorSize { get; set; }
            public bool HasCancer { get; set; }
        }

        public void Run()
        {
            var device = DeviceDescriptor.UseDefaultDevice();

            // 1. Generate Data
            int sampleSize = 32;
            int nbDimensionsInput = 2;// 2 dimensions (age&tumorsize)
            int nbLabels = 2; // l'output est un vecteur de probabilités qui doit sommer à 1. Si on ne met qu'une seule dimension de sortie, l'output sera toujours de 1.
            // on met donc deux dimension, une dimension 'vrai' et une dimension 'faux'. L'output sera du genre 0.25 vrai et 0.75 faux => total des poids = 1;
            // premier label = faux, second = vrai

            IEnumerable<DataPoint> data = GenerateData(sampleSize);

            //foreach (var pt in data)
            //    Debug.WriteLine($"{pt.Age};{pt.TumorSize};{(pt.HasCancer ? 1 : 0)}");

            Variable inputVariables = Variable.InputVariable(NDShape.CreateNDShape(new[] { nbDimensionsInput }), DataType.Double, "input");
            Variable expectedOutput = Variable.InputVariable(new int[] { nbLabels }, DataType.Double, "output");

            Parameter bias = new Parameter(NDShape.CreateNDShape(new[] { nbLabels }), DataType.Double, 0); // une abscisse pour chaque dimension
            Parameter weights = new Parameter(NDShape.CreateNDShape(new[] { nbDimensionsInput, nbLabels }), DataType.Double, 0); // les coefficients à trouver
            // 2 variable d'input, 2 estimations en sortie (proba vrai et proba faux)

            Function predictionFunction = CNTKLib.Plus(CNTKLib.Times(weights, inputVariables), bias);

            Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(predictionFunction, expectedOutput);
            Function evalErrorFunction = CNTKLib.ClassificationError(predictionFunction, expectedOutput);
            //Function logisticClassifier = CNTKLib.Sigmoid(evaluationFunction, "LogisticClassifier");
            uint minibatchSize = 25;
            //double learningRate = 0.5;
            //TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(learningRate, minibatchSize);

            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.3, (uint) (data.Count()/1.0));
            TrainingParameterScheduleDouble momentumSchedule = new TrainingParameterScheduleDouble(0.9126265014311797, minibatchSize);

            var parameters = new ParameterVector();
            foreach (var p in predictionFunction.Parameters())
                parameters.Add(p);

            List<Learner> parameterLearners = new List<Learner>() { CNTKLib.FSAdaGradLearner(parameters, learningRatePerSample, momentumSchedule, true) };

            Trainer trainer = Trainer.CreateTrainer(predictionFunction, lossFunction, evalErrorFunction, parameterLearners);

            double nbSamplesToUseForTraining = 20000;
            int numMinibatchesToTrain = (int)(nbSamplesToUseForTraining / (int)minibatchSize);

            // train the model
            for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
            {
                IEnumerable<DataPoint> trainingData = GenerateData((int)minibatchSize);

                List<double> minibatchInput = new List<double>();
                List<double> minibatchOutput = new List<double>();
                foreach (DataPoint row in trainingData)
                {
                    minibatchInput.Add(row.Age);
                    minibatchInput.Add(row.TumorSize);
                    minibatchOutput.Add(row.HasCancer ? 0d : 1d);
                    minibatchOutput.Add(row.HasCancer ? 1d : 0d);
                }


                Value inputData = Value.CreateBatch<double>(NDShape.CreateNDShape(new int[] { nbDimensionsInput }), minibatchInput, device);
                Value outputData = Value.CreateBatch<double>(NDShape.CreateNDShape(new int[] { nbLabels }), minibatchOutput, device);

                trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { inputVariables, inputData }, { expectedOutput, outputData } }, device);

                PrintTrainingProgress(trainer, minibatchCount);
            }

            // test
            {
                int testSize = 100;
                IEnumerable<DataPoint> trainingData = GenerateData(testSize);

                List<double> minibatchInput = new List<double>();
                List<double> minibatchOutput = new List<double>();
                foreach (DataPoint row in trainingData)
                {
                    minibatchInput.Add(row.Age);
                    minibatchInput.Add(row.TumorSize);
                    minibatchOutput.Add(row.HasCancer ? 0d : 1d);
                    minibatchOutput.Add(row.HasCancer ? 1d : 0d);
                }


                Value inputData = Value.CreateBatch<double>(NDShape.CreateNDShape(new int[] { nbDimensionsInput }), minibatchInput, device);
                Value outputData = Value.CreateBatch<double>(NDShape.CreateNDShape(new int[] { nbLabels }), minibatchOutput, device);

                IList<IList<double>> expectedOneHot = outputData.GetDenseData<double>(predictionFunction.Output);
                IList<int> expectedLabels = expectedOneHot.Select(l => l.IndexOf(1.0d)).ToList();

                var outputDataMap = new Dictionary<Variable, Value>() { { predictionFunction.Output, null } };
                predictionFunction.Evaluate(
                    new Dictionary<Variable, Value>() { { inputVariables, inputData } },
                    outputDataMap,
                    device);

                Value outputValue = outputDataMap[predictionFunction.Output];

                IList<IList<double>> actualLabelSoftMax = outputValue.GetDenseData<double>(predictionFunction.Output);
                var actualLabels = actualLabelSoftMax.Select((IList<double> l) => l.IndexOf(l.Max())).ToList();
                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                Debug.WriteLine($"Validating Model: Total Samples = {testSize}, Misclassify Count = {misMatches}");
            }
        }

        public void PrintTrainingProgress(Trainer trainer, int minibatchIdx)
        {
            if (trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                Debug.WriteLine($"Minibatch: {minibatchIdx} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
            }
        }
        IEnumerable<DataPoint> GenerateData(int sampleSize)
        {
            var random = new Random(0);

            for (int i = 0; i < sampleSize; i++)
            {
                double age = NextGaussian(random) + 3;
                double tumorSize = NextGaussian(random) + 3;
                bool hasCancer = random.Next(2) == 0;
                if (hasCancer)
                {
                    age *= 2;
                    tumorSize *= 2;
                }

                yield return new DataPoint
                {
                    Age = age,
                    TumorSize = tumorSize,
                    HasCancer = hasCancer
                };
            }
        }

        double NextGaussian(Random r)
        {
            double x1 = r.NextDouble();
            double x2 = r.NextDouble();
            double gaussian = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Sin(2.0 * Math.PI * x2);
            return gaussian;
        }

    }
}
