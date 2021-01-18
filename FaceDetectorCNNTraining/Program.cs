using Keras.Layers;
using Keras.Models;
using Numpy;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace FaceDetectorCNNTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            //NDarray images = null;
            //NDarray results = null;

            //LoadNPData(ref images, ref results);

            //results = Keras.Utils.Util.ToCategorical(results, 2);

            var seq = CreateCNN();
            seq.LoadWeight("FaceDetectorCNN.h5");
            seq.Summary();
            //var seq = TrainCNN(images, results);

            //TestWithGlasses(seq);
            SpeedTest(seq);
            //Test(seq, "B:/Desktop/testResults.bin", "B:/Desktop/test/");

            Console.ReadLine();
        }

        private static Sequential CreateCNN()
        {
            var seq = new Sequential();

            // 1 layer
            seq.Add(new Conv2D(32, new Tuple<int, int>(3, 3), activation: "relu", input_shape: new Keras.Shape(19, 19, 1)));
            seq.Add(new MaxPooling2D(new Tuple<int, int>(2, 2)));

            // 2 layer
            seq.Add(new Conv2D(64, new Tuple<int, int>(3, 3), activation: "relu"));
            seq.Add(new MaxPooling2D(new Tuple<int, int>(2, 2)));

            seq.Add(new Flatten());

            // Fully-connected layer
            seq.Add(new Dense(219, activation: "relu"));

            // Output layer
            seq.Add(new Dense(2, activation: "softmax"));
            

            seq.Compile(new Keras.Optimizers.SGD(lr: 0.001f), "categorical_crossentropy", new string[] { "accuracy" });

            return seq;
        }

        private static Sequential TrainCNN(NDarray images, NDarray results)
        {
            var seq = CreateCNN();

            seq.Summary();
            seq.Fit(images, results, batch_size: 32, epochs: 200);

            seq.SaveWeight("FaceDetectorCNN.h5");
            return seq;
        }

        private static void LoadNPData(ref NDarray images, ref NDarray results)
        {
            var bn = new BinaryFormatter();
            var trainResultsFile = new FileStream("B:/Desktop/Results.bin", FileMode.Open);
            var validResutlsFile = new FileStream("B:/Desktop/testResults.bin", FileMode.Open);

            var resultsV = bn.Deserialize(validResutlsFile) as bool[];
            var resultsT = bn.Deserialize(trainResultsFile) as bool[];
            results = np.zeros(resultsT.Length + 10000);

            var filesT = Directory.GetFiles("B:/Desktop/shuffled/");
            var filesV = Directory.GetFiles("B:/Desktop/test/");
            images = np.zeros(resultsT.Length + 10000, 19, 19, 1);

            for (int i = 0; i < filesT.Length; i++)
            {
                var image = Image.Load<Rgba32>("B:/Desktop/shuffled/" + i + ".png");
                image.Mutate(x => x.Resize(19, 19));

                var bitmap = Grayscale(image);

                for (int j = 0; j < image.Height; j++)
                {
                    for (int k = 0; k < image.Width; k++)
                    {
                        images[i][j][k][0] = (NDarray)bitmap[j, k] / 255;
                    }
                }

                results[i] = resultsT[i] ? (NDarray)0 : (NDarray)1;
            }
            for (int i = 0; i < 10000; i++)
            {
                var image = Image.Load<Rgba32>("B:/Desktop/test/" + i + ".png");
                image.Mutate(x => x.Resize(19, 19));

                var bitmap = Grayscale(image);

                for (int j = 0; j < image.Height; j++)
                {
                    for (int k = 0; k < image.Width; k++)
                    {
                        images[filesT.Length + i][j][k][0] = (NDarray)bitmap[j, k] / 255;
                    }
                }

                results[filesT.Length + i] = resultsV[i] ? (NDarray)0 : (NDarray)1;
            }

            trainResultsFile.Close();
            validResutlsFile.Close();
        }

        private static double Test(Sequential seq, string resultsPath, string dataPath)
        {
            bool[] results = null;

            var bn = new BinaryFormatter();
            var resultsFile = new FileStream(resultsPath, FileMode.Open);
            results = bn.Deserialize(resultsFile) as bool[];

            var successes = 0;
            var falseNegatives = 0;
            var falsePositives = 0;
            var trueNegatives = 0;
            var truePositives = 0;
            var detections = 0;

            var files = Directory.GetFiles(dataPath);
            var imagesND = np.zeros(files.Length, 19, 19, 1);

            for (int i = 0; i < files.Length; i++)
            {
                var image = Image.Load<Rgba32>(dataPath + i + ".png");
                image.Mutate(x => x.Resize(19, 19));

                var bitmap = Grayscale(image);

                for (int j = 0; j < image.Height; j++)
                {
                    for (int k = 0; k < image.Width; k++)
                    {
                        imagesND[i][j][k][0] = (NDarray)bitmap[j, k] / 255;
                    }
                }
            }

            var res = seq.PredictOnBatch(imagesND);

            for (int i = 0; i < files.Length; i++)
            {
                var detectionResult = true;
                var detectionScores = np.array(res[i]).GetData<float>();
                if (detectionScores[0] < detectionScores[1]) detectionResult = false;

                if (detectionResult) detections++;
                if (detectionResult && detectionResult == results[i]) truePositives++;
                if (!detectionResult && detectionResult == results[i]) trueNegatives++;
                if (detectionResult == results[i]) successes++;
                else if (!detectionResult) falseNegatives++;
                else falsePositives++;
            }

            Console.WriteLine();
            Console.WriteLine("********************CNN Test:********************");
            Console.WriteLine("Detections: {0}", detections);
            Console.WriteLine("Successful Detection: {0}/{1} - {2}%", successes, results.Length, Math.Round((double)successes / results.Length, 4) * 100);
            Console.WriteLine("False Negatives: {0} - {1}%", falseNegatives, Math.Round((double)falseNegatives / (falseNegatives + truePositives), 4) * 100);
            Console.WriteLine("False Positives: {0} - {1}%", falsePositives, Math.Round((double)falsePositives / (falsePositives + trueNegatives), 4) * 100);
            Console.WriteLine("************************************************");

            resultsFile.Close();

            return Math.Round((double)successes / results.Length, 4) * 100;
        }

        private static void TestWithGlasses(Sequential seq)
        {
            var detections = 0;

            var files = Directory.GetFiles("B:/Desktop/facesWithGlasses");

            var imagesND = np.zeros(files.Length, 19, 19, 1);

            for(var i = 0; i < files.Length; i++)
            {
                var image = Image.Load<Rgba32>(files[i]);
                image.Mutate(x => x.Resize(19, 19));

                var bitmap = Grayscale(image);

                for (int j = 0; j < image.Height; j++)
                {
                    for (int k = 0; k < image.Width; k++)
                    {
                        imagesND[i][j][k][0] = (NDarray)bitmap[j, k] / 255;
                    }
                }
            }

            var res = np.array(seq.PredictOnBatch(imagesND));

            for (int i = 0; i < files.Length; i++)
            {
                var scores = res[i].GetData<float>();
                var detectionResult = true;
                if (scores[0] < scores[1]) detectionResult = false;

                if (detectionResult) detections++;
            }

            Console.WriteLine();
            Console.WriteLine("**********************CNN Glasses Test:**********************");
            Console.WriteLine("Detections: {0}/{1} - {2}%", detections, files.Length, Math.Round((double)detections / files.Length, 4) * 100);
            Console.WriteLine("*********************************************************");
        }

        private static double[,] Grayscale(Image<Rgba32> image)
        {
            var result = new double[image.Height, image.Width];

            for (int i = 0; i < image.Height; i++)
            {
                for (int j = 0; j < image.Width; j++)
                {
                    result[i, j] = (double)(image[j, i].R + image[j, i].G + image[j, i].B) / 3;
                }
            }

            return result;
        }

        private static void SpeedTest(Sequential seq)
        {
            const int N = 1000;

            var image = Image.Load<Rgba32>("B:/Desktop/test/1.png");
            var bitmap = Grayscale(image);

            var imageND = np.zeros(1, 19, 19, 1);
            for (int i = 0; i < 19; i++)
            {
                for (int j = 0; j < 19; j++)
                {
                    imageND[0][i][j][0] = (NDarray)bitmap[i, j];
                }
            }

            var speedResult = 0.0;

            for (int i = 0; i < N; i++)
            {
                var timer = DateTime.Now;

                seq.Predict(imageND, verbose: 0);

                var roundSpeedResult = (DateTime.Now - timer).TotalMilliseconds;

                speedResult += roundSpeedResult;
            }

            

            Console.WriteLine("***********************************Speed Test of CNN***********************************");
            Console.WriteLine("Time: {0}ms", speedResult / N);
            Console.WriteLine("**************************************************************************************");
        }
    }

}
