// MIT Licence (2018)
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class GenericMinibatchSequenceSource
    {
        private readonly int _sequenceLength;
        private readonly Variable _inputVariables;
        private readonly List<IEnumerable<double>> _trainImages;
        private readonly Variable _expectedOutput;
        private readonly List<double[]> _trainLabels1Hot;
        private readonly DeviceDescriptor _device;
        private readonly int _nbMinibatch;
        private int _currentMinibatch;
        private readonly int _numSweepsToTrainWith;
        private readonly uint _minibatchSize;

        private Random _random = new Random(0);
        private List<int> _minibatchIndexes;

        public GenericMinibatchSequenceSource(Variable inputVariables, List<IEnumerable<double>> sequenceOfImages, Variable expectedOutput, List<double[]> trainLabels1Hot, int nbSamplesToUseForTraining, int numSweepsToTrainWith, uint minibatchSize, DeviceDescriptor device)
        {
            _sequenceLength = sequenceOfImages.First().Count();
            _inputVariables = inputVariables;
            _trainImages = sequenceOfImages;
            _expectedOutput = expectedOutput;
            _trainLabels1Hot = trainLabels1Hot;
            _device = device;

            _nbMinibatch = (int)(nbSamplesToUseForTraining * numSweepsToTrainWith / (double)minibatchSize);
            _currentMinibatch = 0;
            _numSweepsToTrainWith = numSweepsToTrainWith;
            _minibatchSize = minibatchSize;

            int nbMinibatchInSample = nbSamplesToUseForTraining / (int)minibatchSize;

            List<int> minibatchIndexes = new List<int>();
            for (int i = 0; i < numSweepsToTrainWith; i++)
            {
                List<int> sweepBatchIndexes = new List<int>();
                for (int j = 0; j < nbMinibatchInSample; j++)
                    sweepBatchIndexes.Add(j);
                sweepBatchIndexes = sweepBatchIndexes.OrderBy(x => _random.NextDouble()).ToList();
                minibatchIndexes.AddRange(sweepBatchIndexes);
            }
            _minibatchIndexes = minibatchIndexes;
        }

        public IDictionary<Variable, MinibatchData> GetNextRandomMinibatch()
        {
            HashSet<int> usedIndexes = new HashSet<int>();
            List<IEnumerable<double>> minibatchInput = new List<IEnumerable<double>>();
            List<double> minibatchOutput = new List<double>();

            for (int i = 0; i < _minibatchSize; i++)
            {
                int imageIndex = _random.Next(_trainImages.Count);
                while (usedIndexes.Contains(imageIndex))
                    imageIndex = _random.Next(_trainImages.Count);

                usedIndexes.Add(imageIndex);

                IEnumerable<double> images = _trainImages[imageIndex];
                double[] labels = _trainLabels1Hot[imageIndex];

                minibatchInput.Add(images);
                minibatchOutput.AddRange(labels.Select(x => (double)x));
            }

            Dictionary<Variable, MinibatchData> data = new Dictionary<Variable, MinibatchData>();

            Value inputData = Value.CreateBatchOfSequences(_inputVariables.Shape, minibatchInput, _device);
            Value outputData = Value.CreateBatch(_expectedOutput.Shape, minibatchOutput, _device);

            data.Add(_inputVariables, new MinibatchData(inputData, _minibatchSize));
            data.Add(_expectedOutput, new MinibatchData(outputData, _minibatchSize));

            _currentMinibatch++;
            if (_currentMinibatch == _minibatchIndexes.Count())
                _currentMinibatch = 0;

            return data;
        }
    }
}
