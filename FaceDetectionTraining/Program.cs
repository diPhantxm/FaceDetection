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
            double[][,] images = null;
            bool[] results = null;
            double[][,] validImages = null;
            bool[] validResults = null;

            var cc = TrainViolaJones(images, results, validImages, validResults);

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

        public static void SaveCascade(CascadeClassifier cc)
        {
            var bn = new BinaryFormatter();
            var classifierFile = new FileStream("CascadeClassifier.bin", FileMode.Create);

            bn.Serialize(classifierFile, cc);
            classifierFile.Close();
        }

        public static CascadeClassifier LoadClassifier()
        {
            var bn = new BinaryFormatter();
            var classifierFile = new FileStream("CascadeClassifier.bin", FileMode.Open);

            var fc = bn.Deserialize(classifierFile) as CascadeClassifier;
            classifierFile.Close();

            return fc;
        }
    }
}
