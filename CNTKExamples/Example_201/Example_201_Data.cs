// MIT Licence (2018)
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Example_201_Item
    {
        public byte Label { get; set; }

        /// <summary>
        /// 32px*32px*3 colors 
        /// The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        /// </summary>
        public byte[] Image { get; set; }
    }

    class Example_201_Data
    {

        public IEnumerable<Example_201_Item> LoadTrainingImages()
        {
            string[] filenames = new[] {
                "./Example_201/Data/data_batch_1.bin",
                "./Example_201/Data/data_batch_2.bin",
                "./Example_201/Data/data_batch_3.bin",
                "./Example_201/Data/data_batch_4.bin",
                "./Example_201/Data/data_batch_5.bin",
            };

            foreach (var filename in filenames)
                foreach (var image in ReadFile(filename))
                    yield return image;
        }

        public IEnumerable<Example_201_Item> LoadTestImages()
        {
            string[] filenames = new[] {
                "./Example_201/test_batch.bin",
            };

            foreach (var filename in filenames)
                foreach (var image in ReadFile(filename))
                    yield return image;
        }

        public IDictionary<byte, string> LoadLabelIndex()
        {
            return ReadLabels("./Example_201/batches.meta.txt");
        }

        public IDictionary<byte, string> ReadLabels(string labelPath)
        {
            FileInfo fi = new FileInfo(labelPath);
            FileStream fs = fi.OpenRead();
            MemoryStream ms = new MemoryStream();
            fs.CopyTo(ms);
            fs.Dispose();

            ms.Seek(0, SeekOrigin.Begin);

            Dictionary<byte, string> rv = new Dictionary<byte, string>();

            using (StreamReader reader = new StreamReader(ms))
            {
                byte i = 0;
                string line;
                while (!string.IsNullOrEmpty(line = reader.ReadLine()))
                {
                    rv[i] = line;
                    i++;
                }
            }

            return rv;
        }

        public IEnumerable<Example_201_Item> ReadFile(string filepath)
        {
            FileInfo trainImagesFile = new FileInfo(filepath);
            FileStream fs = trainImagesFile.OpenRead();
            MemoryStream ms = new MemoryStream();
            fs.CopyTo(ms);
            fs.Dispose();

            byte[] itemBuffer = new byte[3072 + 1];
            ms.Seek(0, SeekOrigin.Begin);

            while(ms.Read(itemBuffer, 0, 3072+1) != 0)
            {
                Example_201_Item item = new Example_201_Item();

                byte[] imageBuffer = new byte[3072];
                Array.Copy(itemBuffer, 1, imageBuffer, 0, imageBuffer.Length);

                item.Label = itemBuffer[0];
                item.Image = imageBuffer;

                yield return item;
            }
        }
    }
}
