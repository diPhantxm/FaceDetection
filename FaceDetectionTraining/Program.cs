using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Drawing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using FaceDetection;
using Newtonsoft.Json;

namespace FaceDetectionTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            //Shuffle("B:/Desktop/Results.bin", "B:/Desktop/shuffled/", "B:/Desktop/faces/", "B:/Desktop/notFaces/");
            //Shuffle("B:/Desktop/testResults.bin", "B:/Desktop/test/", "B:/Desktop/testFaces/", "B:/Desktop/testNotFaces/");

            //double[][,] images = null;
            //bool[] results = null;
            //double[][,] validImages = null;
            //bool[] validResults = null;

            //LoadData(ref images, ref results, "B:/Desktop/Results.bin", "B:/Desktop/shuffled/");
            //LoadData(ref validImages, ref validResults, "B:/Desktop/testResults.bin", "B:/Desktop/test/");

            //validImages = validImages.Take(10000).ToArray();
            //validResults = validResults.Take(10000).ToArray();

            //var cc = TrainViolaJones(images, results, validImages, validResults);
            var cc = LoadClassifier();

            //TestWithGlasses(cc);
            SpeedTest(cc);
            //Test(cc, "B:/Desktop/testResults.bin", "B:/Desktop/test/");


            Console.ReadLine();
        }

        

        public static CascadeClassifier TrainViolaJones(double[][,] trainImgs, bool[] trainResults, double[][,] validImgs, bool[] validResults)
        {
            var features = new List<WeakClassifier>();
            features.AddRange(CreateFeatures(HaarFeatureType.Type1, 19, 19));
            features.AddRange(CreateFeatures(HaarFeatureType.Type2, 19, 19));
            features.AddRange(CreateFeatures(HaarFeatureType.Type3, 19, 19));
            features.AddRange(CreateFeatures(HaarFeatureType.Type4, 19, 19));
            features.AddRange(CreateFeatures(HaarFeatureType.Type5, 19, 19));

            var cc = new CascadeClassifier();
            cc.Train(features, trainImgs, trainResults, 0.5, 0.6, 0.0001, validImgs, validResults);

            SaveCascade(cc);
            return cc;
        }

        public static void Shuffle(string endFile, string destination, string faceFolder, string nonFaceFolder)
        {
            var rnd = new Random();

            var files = Directory.GetFiles(faceFolder).ToList();
            var filesBuffer = Directory.GetFiles(nonFaceFolder);
            
            files.AddRange(filesBuffer);

            var facesCount = files.Count - filesBuffer.Length;
            var notFacesCount = filesBuffer.Length;
            var total = facesCount + notFacesCount;

            var results = new bool[total];
            
            for (int i = 0; i < total; i++)
            {
                var filePath = String.Empty;

                if (rnd.Next() % 2 == 0 && facesCount > 0 || notFacesCount == 0)
                {
                    filePath = Directory.GetFiles(faceFolder)[rnd.Next() % facesCount];
                    facesCount--;
                    results[i] = true;
                }
                else
                {
                    filePath = Directory.GetFiles(nonFaceFolder)[rnd.Next() % notFacesCount];
                    notFacesCount--;
                    results[i] = false;
                }

                File.Copy(filePath, destination + i + ".png");
            }

            var fileStream = new FileStream(endFile, FileMode.Create);

            var bn = new BinaryFormatter();
            bn.Serialize(fileStream, results);

            fileStream.Close();
        }

        public static double[,] Grayscale(Image<Rgba32> image)
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

        public static List<WeakClassifier> CreateFeatures(HaarFeatureType type, int maxHeight, int maxWidth)
        {
            var features = new List<WeakClassifier>();

            var heightBreak = false;
            var widthBreak = false;
            var yBreak = false;

            for (int i = 1; i <= maxHeight; i++)
            {
                for (int j = 1; j <= maxWidth; j++)
                {
                    for (int y = 0; y <= maxHeight - i; y++)
                    {
                        for (int x = 0; x <= maxWidth - j; x++)
                        {
                            try
                            {
                                features.Add(new WeakClassifier((HaarFeatureType)type, i, j, x, y));
                            }
                            catch (ArgumentException ae)
                            {
                                if (ae.ParamName == "width")
                                {
                                    widthBreak = true;
                                    break;
                                }
                                if (ae.ParamName == "height")
                                {
                                    heightBreak = true;
                                    break;
                                }
                            }

                        }

                        if (yBreak)
                        {
                            yBreak = false;
                            continue;
                        }
                        if (widthBreak || heightBreak) break;
                    }

                    if (widthBreak)
                    {
                        widthBreak = false;
                        continue;
                    }
                    if (heightBreak) break;
                }

                if (heightBreak)
                {
                    heightBreak = false;
                    continue;
                }
            }

            return features;
        }

        public static void LoadData(ref double[][,] images, ref bool[] results, string resultsPath, string dataPath)
        {
            var bn = new BinaryFormatter();
            var resultsFile = new FileStream(resultsPath, FileMode.Open);

            results = bn.Deserialize(resultsFile) as bool[];

            var files = Directory.GetFiles(dataPath);
            images = new double[files.Length][,];

            for (int i = 0; i < files.Length; i++)
            {

                var image = Image.Load<Rgba32>(dataPath + i + ".png");
                image.Mutate(x => x.Resize(19, 19));
                images[i] = new double[19, 19];

                var bitmap = Grayscale(image);

                for (int j = 0; j < image.Height; j++)
                {
                    for (int k = 0; k < image.Width; k++)
                    {
                        images[i][j, k] = bitmap[j, k] / 255;
                    }
                }
            }
        }

        public static void SaveClassifier(StrongClassifier fc)
        {
            var bn = new BinaryFormatter();
            var classifierFile = new FileStream("B:\\Desktop\\StrongClassifier.bin", FileMode.Create);

            bn.Serialize(classifierFile, fc);
            classifierFile.Close();
        }

        public static void SaveCascade(CascadeClassifier cc)
        {
            var bn = new BinaryFormatter();
            var classifierFile = new FileStream("B:\\Desktop\\CascadeClassifier.bin", FileMode.Create);

            bn.Serialize(classifierFile, cc);
            classifierFile.Close();
        }

        public static CascadeClassifier LoadClassifier()
        {
            var bn = new BinaryFormatter();
            var classifierFile = new FileStream("B:\\Desktop\\CascadeClassifier.bin", FileMode.Open);

            var fc = bn.Deserialize(classifierFile) as CascadeClassifier;
            classifierFile.Close();

            return fc;
        }

        public static double Test(CascadeClassifier cc, string resultsPath, string dataPath)
        {
            double[][,] images = null;
            bool[] results = null;

            LoadData(ref images, ref results, resultsPath, dataPath);

            var i = 0;
            var successes = 0;
            var falseNegatives = 0;
            var falsePositives = 0;
            var detections = 0;
            var trueNegatives = 0;
            var truePositives = 0;
            foreach (var image in images)
            {
                var classifierResult = cc.Detect(image);

                if (classifierResult) detections++;
                if (classifierResult && classifierResult == results[i]) truePositives++;
                if (!classifierResult && classifierResult == results[i]) trueNegatives++;
                if (classifierResult == results[i++]) successes++;
                else if (!classifierResult) falseNegatives++;
                else falsePositives++;
            }

            Console.WriteLine();
            Console.WriteLine("************Cascade Classifier Test:************");
            Console.WriteLine("Detections: {0}", detections);
            Console.WriteLine("Successful Detection: {0} - {1}%", successes, Math.Round((double)successes / results.Length, 4) * 100);
            Console.WriteLine("False Negatives: {0} - {1}%", falseNegatives, Math.Round((double)falseNegatives/ (falseNegatives + truePositives), 4) * 100);
            Console.WriteLine("False Positives: {0} - {1}%", falsePositives, Math.Round((double)falsePositives / (falsePositives + trueNegatives), 4) * 100);
            Console.WriteLine("************************************************");

            return Math.Round((double)successes / results.Length, 4) * 100;
        }

        private static void TestWithGlasses(CascadeClassifier cc)
        {
            var detections = 0;

            var files = Directory.GetFiles("B:/Desktop/facesWithGlasses");

            for (int i = 0; i < files.Length; i++)
            {
                var image = Image.Load<Rgba32>(files[i]);
                var bitmap = Grayscale(image);

                if (cc.Detect(bitmap)) detections++;
            }

            Console.WriteLine();
            Console.WriteLine("**********************Viola Jones Glasses Test:**********************");
            Console.WriteLine("Detections: {0}/{1} - {2}", detections, files.Length, Math.Round((double)detections / files.Length, 4) * 100);
            Console.WriteLine("**********************************************************************");
        }

        private static void SpeedTest(CascadeClassifier cc)
        {
            double[][,] images = null;
            bool[] results = null;

            LoadData(ref images, ref results, "B:/Desktop/testResults.bin", "B:/Desktop/test/");

            var passedEach = new int[cc.Classifiers.Count];
            var timeEach = new double[cc.Classifiers.Count];

            for (int i = 0; i < images.Length; i++)
            {
                var rejected = cc.RejectedOn(images[i]);
                if (rejected != -1)
                {
                    for (int j = 0; j < rejected + 1; j++)
                    {
                        passedEach[j]++;
                    }
                }
                else passedEach[0]++;
            }

            var integrals = cc.IntegrateImages(images);

            for (int i = 0; i < cc.Classifiers.Count; i++)
            {
                var timer = DateTime.Now;

                for (int j = 0; j < integrals.Length; j++)
                {
                    cc.Classifiers[i].Detect(integrals[j]);
                }

                timeEach[i] = (DateTime.Now - timer).TotalMilliseconds / integrals.Length;
            }

            var speedResult = 0.0;
            for (int i = 0; i < cc.Classifiers.Count; i++)
            {
                speedResult += timeEach[i] * (double)passedEach[i] / images.Length;
            }

            Console.WriteLine("***************************Speed Test of Cascade Classifier***************************");
            Console.WriteLine("Time: {0}ms", speedResult);
            Console.Write("Time on Each: ");
            for (int i = 0; i < cc.Classifiers.Count; i++)
            {
                Console.Write("{0} ", Math.Round(timeEach[i], 3));
            }
            Console.WriteLine();
            Console.Write("Images passed on each: ");
            for (int i = 0; i < cc.Classifiers.Count; i++)
            {
                Console.Write("{0} ", Math.Round((double)passedEach[i] / images.Length, 4));
            }
            Console.WriteLine();
            Console.WriteLine("**************************************************************************************");
        }
    }
}
