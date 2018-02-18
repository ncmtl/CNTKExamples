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
    class Example_103_Util
    {
        public int[] ConvertTo1Hot(int label0To9)
        {
            int[] output = new int[10];
            output[label0To9] = 1;
            return output;
        }

        public void AsciiPrint(byte[] image)
        {
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                {
                    byte value = image[i * 28 + j];
                    if (value == 0)
                        Debug.Write(" ");
                    else
                        Debug.Write("*");
                    if (j == 27)
                        Debug.WriteLine("");
                }
            // throw new NotImplementedException();
        }

        public List<byte[]> LoadImages(string imagePackPath)
        {
            FileInfo trainImagesFile = new FileInfo(imagePackPath);
            GZipStream gZipStream = new GZipStream(trainImagesFile.OpenRead(), CompressionMode.Decompress);
            MemoryStream ms = new MemoryStream();
            gZipStream.CopyTo(ms);
            gZipStream.Dispose();

            byte[] buffer4bytes = new byte[4];
            ms.Seek(0, SeekOrigin.Begin);

            ms.Read(buffer4bytes, 0, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(buffer4bytes);
            int magicNumber = BitConverter.ToInt32(buffer4bytes, 0);

            ms.Read(buffer4bytes, 0, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(buffer4bytes);
            int nbImages = BitConverter.ToInt32(buffer4bytes, 0);

            ms.Read(buffer4bytes, 0, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(buffer4bytes);
            int nbRow = BitConverter.ToInt32(buffer4bytes, 0);

            ms.Read(buffer4bytes, 0, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(buffer4bytes);
            int nbCols = BitConverter.ToInt32(buffer4bytes, 0);

            List<byte[]> images = new List<byte[]>();
            for (int i = 0; i < nbImages; i++)
            {
                byte[] imageBuffer = new byte[nbRow * nbCols];
                ms.Read(imageBuffer, 0, nbRow * nbCols);
                images.Add(imageBuffer);
            }

            return images;
        }
        
        public List<int> LoadLabels(string labelPackPath)
        {
            FileInfo trainImagesFile = new FileInfo(labelPackPath);
            GZipStream gZipStream = new GZipStream(trainImagesFile.OpenRead(), CompressionMode.Decompress);
            MemoryStream ms = new MemoryStream();
            gZipStream.CopyTo(ms);
            gZipStream.Dispose();

            byte[] buffer4bytes = new byte[4];
            ms.Seek(0, SeekOrigin.Begin);

            ms.Read(buffer4bytes, 0, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(buffer4bytes);
            int magicNumber = BitConverter.ToInt32(buffer4bytes, 0);

            ms.Read(buffer4bytes, 0, 4);
            if (BitConverter.IsLittleEndian) Array.Reverse(buffer4bytes);
            int nbLabels = BitConverter.ToInt32(buffer4bytes, 0);

            List<int> labels = new List<int>();
            for (int i = 0; i < nbLabels; i++)
            {
                int label = ms.ReadByte();
                labels.Add(label);
            }

            return labels;
        }
        
        public Function DenseNode(Variable input, int outputDim, Func<Variable, Function> activation)
        {
            var linearLayer = LinearNode(input, outputDim);
            if (activation == null) return linearLayer;
            return activation(linearLayer);
        }

        public Function LinearNode(Variable input, int outputDim)
        {
            if (input.Shape.Rank != 1)
            {
                // un dense layer prend un vecteur seulement en entrée => on transforme l'entrée en vecteur
                int nbDims = 1;
                for (int i = 0; i < input.Shape.Dimensions.Count; i++)
                    nbDims = nbDims * input.Shape.Dimensions[i];

                var vector = CNTKLib.Reshape(input, NDShape.CreateNDShape(new[] { nbDims }));
                return LinearNode(vector, outputDim);
            }
            
            List<int> weightDimensions = new List<int>();
            weightDimensions.Add(outputDim);
            weightDimensions.Add(input.Shape.Dimensions.Single());
            
            Parameter weights = new Parameter(NDShape.CreateNDShape(weightDimensions), DataType.Double,
                CNTKLib.GlorotUniformInitializer( // valeur par défaut aléatoire
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1)); // les coefficients à trouver

            Parameter bias = new Parameter(NDShape.CreateNDShape(new[] { outputDim }), DataType.Double, 0); // une abscisse pour chaque dimension

            Function ax = CNTKLib.Times(weights, input);
            Function ax_plus_b = CNTKLib.Plus(bias, ax);
            return ax_plus_b;
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
        
        Parameter Parameter(IEnumerable<int> dims)
        {
            return new Parameter(NDShape.CreateNDShape(dims), DataType.Double,
               CNTKLib.GlorotUniformInitializer( // valeur par défaut aléatoire
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1));
        }

        internal Function ConvolutionWithBatchNormalization(Variable input, int[] filterSize, int nbFilter, Func<Variable, Function> activation)
        {
            List<int> stridesDim = new List<int>();
            for (int i = 0; i < input.Shape.Dimensions.Count - 1; i++)
                stridesDim.Add(1);
            return ConvolutionWithBatchNormalization(input, filterSize, nbFilter, stridesDim.ToArray(), activation);
        }
        
        internal Function ConvolutionWithBatchNormalization(Variable input, int[] filterSize, int nbFilter, int[] strides, Func<Variable, Function> activation)
        {
            var net = Convolution(input, filterSize, nbFilter, strides, null);
            net = BatchNormalization(net);
            if (activation == null) return net;
            return activation(net);            
        }

        internal Function Convolution(Variable input, int[] filterSize, int nbFilter, Func<Variable, Function> activation)
        {
            List<int> stridesDim = new List<int>();
            for (int i = 0; i < input.Shape.Dimensions.Count - 1; i++)
                stridesDim.Add(1);
            return Convolution(input, filterSize, nbFilter, stridesDim.ToArray(), activation);
        }
        
        internal Function Convolution(Variable input, int[] filterSize, int nbFilter, int[] strides, Func<Variable, Function> activation)
        {
            int numInputChannels = input.Shape.Dimensions.Last();
            
            List<int> kernelDims = new List<int>();
            kernelDims.AddRange(filterSize);
            kernelDims.Add(numInputChannels);
            kernelDims.Add(nbFilter);
            
            List<int> strideDims = new List<int>();
            strideDims.AddRange(strides);
            strideDims.Add(numInputChannels);

            var convParams = Parameter(kernelDims);
            var conv = CNTKLib.Convolution(convParams, input, NDShape.CreateNDShape(strideDims));

            if (activation == null) return conv;
            return activation(conv);
        }

        //internal Function Convolution(Variable input, int[] filterSize, int nbFilter, int[] stridesDim, Func<Variable, Function> activation)
        //{
        //    List<int> convoParameterDims = new List<int>();
        //    convoParameterDims.AddRange(filterSize);
        //    convoParameterDims.Add(nbFilter);

        //    var filter = new Parameter(NDShape.CreateNDShape(convoParameterDims), DataType.Double,
        //       CNTKLib.GlorotUniformInitializer( // valeur par défaut aléatoire
        //            CNTKLib.DefaultParamInitScale,
        //            CNTKLib.SentinelValueForInferParamInitRank,
        //            CNTKLib.SentinelValueForInferParamInitRank, 1));
        //    var conv = CNTKLib.Convolution(filter, input, stridesDim);
        //    return activation(conv);
        //}

        internal Function Pooling(Variable input, int[] filterSize, int nbLayers, int[] stridesDim)
        {
            var conv = CNTKLib.Pooling(input, PoolingType.Max, filterSize, stridesDim);
            return conv;
        }

        internal Function BatchNormalization(Function layer1)
        {
            var dims = layer1.Output.Shape.Dimensions;
            Variable scale = Parameter(dims);
            Variable bias = Parameter(dims);
            Variable runningMean = Parameter(dims);
            Variable runningInvStd = Parameter(dims);
            Variable runningCount = Parameter(new[] { 1 });

            var bn = CNTKLib.BatchNormalization(layer1, scale, bias, runningMean, runningInvStd, runningCount, spatial:true);
            return bn;
        }

        internal Function PoolingMax(Variable input, int[] filterSize, int[] stridesDim)
        {
            var conv = CNTKLib.Pooling(input, PoolingType.Max, new[] { 1, 1 }, stridesDim);
            return conv;
        }

        internal Function PoolingAvg(Variable input, int[] filterSize, int[] stridesDim)
        {
            var conv = CNTKLib.Pooling(input, PoolingType.Average, new[] { 1, 1 }, stridesDim);
            return conv;
        }


        public class LSTM
        {
            public Function H_output { get; set; }
            public Function C_cellstate { get; set; }
        }


        internal Variable ResNetBasicStack(Variable input, int nbFilters, int stackSize, Func<Variable, Function> activation)
        {
            var net = input;
            for(int i = 0; i < stackSize; i++)            
                net = ResNetBasic(net, nbFilters, activation);

            return net;
        }

        internal Variable ResNetBasic(Variable input, int nbFilters, Func<Variable, Function> activation)
        {
            var net = input;
            net = ConvolutionWithBatchNormalization(net, new[] { 3, 3 }, nbFilters, activation);
            net = ConvolutionWithBatchNormalization(net, new[] { 3, 3 }, nbFilters, null);

            var p = CNTKLib.Plus(net, input);
            if (activation == null) return p;
            return activation(p);            
        }

        internal Variable ResNetInc(Variable input, int nbFilters, Func<Variable, Function> activation)
        {
            var net = input;
            net = ConvolutionWithBatchNormalization(net, new[] { 3, 3 }, nbFilters, new[] { 2, 2 }, activation);
            net = ConvolutionWithBatchNormalization(net, new[] { 3, 3 }, nbFilters, null);

            var s = ConvolutionWithBatchNormalization(input, new[] { 1, 1 }, nbFilters, new[] { 2, 2 }, null);

            var p = CNTKLib.Plus(net, s);
            if (activation == null) return p;
            return activation(p);
        }

        /// <summary>
        /// Implementation of : http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        /// </summary>
        public LSTM LSTMNode(Variable input, int cellDims)
        {
            int outputDim = cellDims;

            Variable previousC = CNTKLib.PlaceholderVariable(NDShape.CreateNDShape(new int[] { cellDims }), DataType.Double, "previousC", new AxisVector(input.DynamicAxes.ToList())); // placeholder for graph building
            Variable previousH = CNTKLib.PlaceholderVariable(NDShape.CreateNDShape(new int[] { outputDim }), DataType.Double, "previousH", new AxisVector(input.DynamicAxes.ToList())); // placeholder for graph building

            Variable forgetGate = DenseNode(CNTKLib.Plus(LinearNode(input, cellDims), LinearNode(previousH, cellDims)), cellDims, CNTKLib.Sigmoid);
            Variable inputGate = DenseNode(CNTKLib.Plus(LinearNode(input, cellDims), LinearNode(previousH, cellDims)), cellDims, CNTKLib.Sigmoid);
            Variable partialC = DenseNode(CNTKLib.Plus(LinearNode(input, cellDims), LinearNode(previousH, cellDims)), cellDims, CNTKLib.Sigmoid);

            Function filteredOldState = CNTKLib.ElementTimes(forgetGate, previousC); // element wise multiplication https://www.mathworks.com/help/matlab/ref/times.html
            Function filteredNewState = CNTKLib.ElementTimes(inputGate, partialC);
            Function c = CNTKLib.Plus(filteredOldState, filteredNewState);

            Function outputGate = DenseNode(CNTKLib.Plus(LinearNode(input, cellDims), LinearNode(previousH, cellDims)), outputDim, CNTKLib.Sigmoid);
            Function h = CNTKLib.ElementTimes(outputGate, c);

            // replace placeholders by recursive variables
            h.ReplacePlaceholders(new Dictionary<Variable, Variable>
            {
                { previousH, CNTKLib.PastValue(h) },
                { previousC, CNTKLib.PastValue(c) },
            });

            return new LSTM
            {
                H_output = h,
                C_cellstate = c
            };
        }

    }
}
