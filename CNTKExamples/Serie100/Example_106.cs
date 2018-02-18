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
    class Example_106_Data
    {
        public List<double> Observations { get; set; }
        public double ExpectedPrediction { get; set; }
    }

    class Example_106
    {
        private Random _random;
        private Example_103_Util _util;

        public Example_106()
        {
            _random = new Random(0);
            _util = new Example_103_Util();
        }

        List<Example_106_Data> GenerateExampleData(int nbItems, int nbObservationsInItem, int nbStepsAhead)
        {
            List<Example_106_Data> rv = new List<Example_106_Data>();

            List<double> dataSeries = GenerateDataSeries(nbItems + nbObservationsInItem + nbStepsAhead).ToList();

            for(int i = 0; i < nbItems; i++)
            {
                Example_106_Data data = new Example_106_Data();
                data.Observations = dataSeries.Skip(i).Take(nbObservationsInItem).ToList();
                data.ExpectedPrediction = dataSeries[i + nbObservationsInItem + (nbStepsAhead - 1)];
                rv.Add(data);
            }

            return rv;
        }

        IEnumerable<double> GenerateDataSeries(int nbObservations)
        {
            for (int i = 0; i < nbObservations; i++)
                yield return Math.Sin(i / 100.0);
        }


        public void Run()
        {
            var device = DeviceDescriptor.UseDefaultDevice();

            int nbObservationsInItem = 5;
            List<Example_106_Data> allData = GenerateExampleData(10000, nbObservationsInItem, 5);
            List<Example_106_Data> trainingData = allData.Take(8000).ToList();
            List<Example_106_Data> evalData = allData.Skip(8000).ToList();

            int[] inputDim = new int[] { 1 };
            Variable input = Variable.InputVariable(NDShape.CreateNDShape(inputDim), DataType.Double, "input", dynamicAxes:new List<Axis>() { Axis.DefaultDynamicAxis(), Axis.DefaultBatchAxis() }); // default dynamic axis ???
            int outputDim = 1;

            Function model = DefineModel_LSTM(input, nbObservationsInItem, outputDim);

            Function output = model.Output;

            //Function output = Variable.InputVariable(NDShape.CreateNDShape(new[] { 1 }), DataType.Float, "output", model.Output.DynamicAxes);

            //IEnumerable<int> inputSequence = ...; // pas clair
            //Value sequence = Value.CreateSequence(NDShape.CreateNDShape(new int[] { 1 }), inputSequence, device);

            uint minibatchSize = 100;

            Variable expectedOutput = Variable.InputVariable(NDShape.CreateNDShape(new int[] { outputDim }), DataType.Double, "expectedOutput", dynamicAxes: new List<Axis>() { Axis.DefaultBatchAxis() }); // default dynamic axis ???
            Trainer trainer = MakeTrainer(expectedOutput, output, model, minibatchSize);

            {   // train
                int nbSamplesToUseForTraining = trainingData.Count;
                int numSweepsToTrainWith = 20;
                int numMinibatchesToTrain = nbSamplesToUseForTraining * numSweepsToTrainWith / (int)minibatchSize;
                var trainingInput = trainingData.Select(x => x.Observations.Select(y => y)).ToList();
                var trainingOutput = trainingData.Select(x => new[] { x.ExpectedPrediction }).ToList();
                var trainingMinibatchSource = new GenericMinibatchSequenceSource(input, trainingInput, expectedOutput, trainingOutput, nbSamplesToUseForTraining, numSweepsToTrainWith, minibatchSize, device);
                RunTraining(trainer, trainingMinibatchSource, numMinibatchesToTrain, device);
            }

            // evaluate
            Evaluate(model, evalData, input, device);            
        }

        private void Evaluate(Function model, List<Example_106_Data> evalData, Variable input, DeviceDescriptor device)
        {
            Debug.WriteLine($"prediction;expectedPrediction;");
            foreach (Example_106_Data data in evalData)
            {
                double prediction = Evaluate(model, data, input, device);
                Debug.WriteLine($"{prediction:n6};{data.ExpectedPrediction};");
            }
        }

        private double Evaluate(Function model, Example_106_Data data, Variable input, DeviceDescriptor device)
        {
            Value inputData = Value.CreateSequence<double>(input.Shape, data.Observations, device);

            var inputDic = new Dictionary<Variable, Value>() { { input, inputData } };
            var outputDic = new Dictionary<Variable, Value>() { { model.Output, null } };

            model.Evaluate(inputDic,outputDic,device);

            Value outputValue = outputDic[model.Output];

            IList<IList<double>> prediction = outputValue.GetDenseData<double>(model.Output);

            return prediction.Single().Single();
        }

        private void RunTraining(Trainer trainer, GenericMinibatchSequenceSource minibatchSource, int numMinibatchesToTrain, DeviceDescriptor device)
        {           
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

        private Trainer MakeTrainer(Variable expectedOutput, Variable output, Function model, uint minibatchSize)
        {
            double learningRate = 0.02;
            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(learningRate);
            TrainingParameterScheduleDouble momentumSchedule = new TrainingParameterScheduleDouble(0.9, minibatchSize);

            Function lossFunction = CNTKLib.SquaredError(expectedOutput, output);
            Function evalErrorFunction = CNTKLib.SquaredError(expectedOutput, output);


            var parameters = new ParameterVector();
            foreach (var p in model.Parameters())
                parameters.Add(p);

            List<Learner> parameterLearners = new List<Learner>() { CNTKLib.FSAdaGradLearner(parameters, learningRatePerSample, momentumSchedule, true) };
            Trainer trainer = Trainer.CreateTrainer(model, lossFunction, evalErrorFunction, parameterLearners);

            return trainer;            
        }

        Function DefineModel_LSTM(Variable input, int memoryLengthDim, int outputDims)
        {
            var lstm = _util.LSTMNode(input, memoryLengthDim);
            Function sequenceLast = CNTKLib.SequenceLast(lstm.H_output);
            Function dropout = CNTKLib.Dropout(sequenceLast, 0.2, 1);
            Function dense = _util.DenseNode(dropout, outputDims, null);
            return dense;
        }

    }
}
