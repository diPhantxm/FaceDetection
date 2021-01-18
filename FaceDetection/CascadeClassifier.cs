using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.IO;

namespace FaceDetection
{
    [Serializable]
    public class CascadeClassifier
    {
        public List<StrongClassifier> Classifiers { get; private set; }

        public CascadeClassifier()
        {

        }

        public CascadeClassifier(IEnumerable<StrongClassifier> classifiers)
        {
            Classifiers = classifiers.ToList();
        }



        public void AddClassifier(StrongClassifier classifier)
        {
            if (classifier.WeakClassifiers.Count == 0) throw new ArgumentException("Classifier cannot have 0 features", nameof(classifier.WeakClassifiers));

            Classifiers.Add(classifier);
        }

        public void AddClassifier(List<StrongClassifier> classifiers)
        {
            for (int i = 0; i < classifiers.Count; i++)
            {
                AddClassifier(classifiers[i]);
            }
        }

        public int RejectedOn(double[,] image)
        {
            for (int i = 0; i < Classifiers.Count; i++)
            {
                if (!Classifiers[i].Detect(image))
                {
                    return i;
                }
            }

            return -1;
        }

        public bool Detect(double[,] image)
        {
            for (int i = 0; i < Classifiers.Count; i++)
            {
                if (!Classifiers[i].Detect(image)) return false;
            }

            return true;
        }

        /// <summary>
        /// Viola Jones Cascade Classifier Training Method
        /// </summary>
        /// <param name="features">Haar-Features</param>
        /// <param name="images">Image data</param>
        /// <param name="results">Results data</param>
        /// <param name="f">Maximum false positive rate</param>
        /// <param name="d">Minimum detection rate</param>
        /// <param name="fTarget">Target overall false positive rate</param>
        public void Train(List<WeakClassifier> features, double[][,] images, bool[] results, double f, double d, double fTarget, double[][,] validImages, bool[] validResults)
        {
            Classifiers = new List<StrongClassifier>();
            var imagesCpy = images.Clone() as double[][,]; // copy not to damage original images
            var resultsCpy = results.Clone() as bool[]; // copy not to damage original results

            var integralValidImages = IntegrateImages(validImages);

            var falsePositiveRates = new List<double>();
            falsePositiveRates.Add(1.0);

            var detectionRates = new List<double>();
            detectionRates.Add(1.0);

            var i = 0;

            Console.WriteLine("*********Traning of the cascade classifier started*********");

            // Train Cascade Classifier
            while (falsePositiveRates[i] > fTarget)
            {
                // Strong classifiers counter
                i++;

                // WeakClassifiers counter
                var n = 0;
                falsePositiveRates.Add(falsePositiveRates.Last());
                detectionRates.Add(0.0);

                // Train Strong Classifier
                var sc = new StrongClassifier();
                while (falsePositiveRates[i] > f * falsePositiveRates[i - 1])
                {
                    n++;

                    Console.Write("    ");

                    // Train Strong Classifier with n Weak Classifiers
                    sc.AddWeakClassifier(imagesCpy, resultsCpy, features);
                    Classifiers.Add(sc);

                    var detectionRate = 0.0;
                    var falsePositiveRate = 0.0;

                    // Evaluate current cascade classifier to determine detection rate and false positives rate
                    Evaluate(integralValidImages, validResults, out falsePositiveRate, out detectionRate);

                    detectionRates[i] = detectionRate;
                    falsePositiveRates[i] = falsePositiveRate;

                    while (detectionRate < d * detectionRates[i - 1])
                    {
                        // Decrease Threshold by 0.001
                        sc.DecreaseThreshold();

                        Evaluate(integralValidImages, validResults, out falsePositiveRate, out detectionRate);

                        detectionRates[i] = detectionRate;
                        falsePositiveRates[i] = falsePositiveRate;
                    }

                    Classifiers.Remove(sc);
                }

                Classifiers.Add(sc);

                Console.WriteLine("Strong classifier was added. Number of weak classifiers: {0} Detection Rate: {1} False Positive Rate: {2}", 
                    sc.WeakClassifiers.Count.ToString().PadRight(10),
                    Math.Round(detectionRates[i], 4).ToString().PadRight(10), 
                    Math.Round(falsePositiveRates[i], 4).ToString().PadRight(10));

                // Get images that were classified incorrectly
                var posImages = new List<double[,]>();
                var posResults = new List<bool>();
                for (int j = 0; j < imagesCpy.Length; j++)
                {
                    var classifierResult = sc.Detect(imagesCpy[j]);

                    if (classifierResult)
                    {
                        posImages.Add(imagesCpy[j]);
                        posResults.Add(resultsCpy[j]);
                    }
                }

                imagesCpy = posImages.ToArray();
                resultsCpy = posResults.ToArray();

                if (falsePositiveRates[i] > fTarget)
                {
                    var fDetImgs = new List<double[,]>();
                    var fDetResults = new List<bool>();

                    for (int j = 0; j < validImages.Length; j++)
                    {
                        var result = Detect(validImages[j]);

                        if (result != validResults[j] && result)
                        {
                            fDetImgs.Add(validImages[j]);
                            fDetResults.Add(validResults[j]);
                        }
                    }

                    var toTakeCount = new int[] { resultsCpy.Count(r => r) - resultsCpy.Count(r => !r), fDetImgs.Count }.Where(x => x >= 0).Min();

                    fDetImgs = fDetImgs.Take(toTakeCount).ToList();
                    fDetImgs.AddRange(imagesCpy);
                    imagesCpy = fDetImgs.ToArray();

                    fDetResults = fDetResults.Take(toTakeCount).ToList();
                    fDetResults.AddRange(resultsCpy);
                    resultsCpy = fDetResults.ToArray();
                }
            }

            Console.WriteLine("*********Traning of the cascade classifier ended*********");

            Statistics(images, results);
        }

        public void Evaluate(double[][,] integralImages, bool[] results, out double fPositiveRate, out double detectionRate)
        {
            if (integralImages.Length != results.Length) throw new ArgumentException("Images legnth does not equal to results length.");

            var falsePositives = 0;
            var detections = 0;

            for (int i = 0; i < integralImages.Length; i++)
            {
                var result = true;

                for (int j = 0; j < Classifiers.Count; j++)
                {
                    if (!Classifiers[j].DetectOnIntegral(integralImages[i]))
                    {
                        result = false;
                        break;
                    }
                }

                if (result) detections++;
                if (result && !results[i]) falsePositives++;
            }

            fPositiveRate = (double)falsePositives / results.Count(r => !r);
            detectionRate = (double)detections / integralImages.Length;
        }

        public void Statistics(double[][,] images, bool[] results)
        {
            var success = 0;
            var positives = 0;
            var falsePositives = 0;
            var falseNegatives = 0;
            for (int i = 0; i < images.GetLength(0); i++)
            {
                var detectionResult = Detect(images[i]);

                if (detectionResult) positives++;
                if (detectionResult == results[i]) success++;
                if (detectionResult != results[i] && detectionResult) falsePositives++;
                if (detectionResult != results[i] && !detectionResult) falseNegatives++;
            }

            using (var file = new StreamWriter("B:\\Desktop\\CascadeStatisticss.txt", append: true))
            {
                file.WriteLine("Detections:\t\t\t {0}\t{1}%", positives, Math.Round((double)positives / results.ToList().Count(r => r == true), 2) * 100);
                file.WriteLine("Successful detections:\t\t {0}\t{1}%", success, Math.Round((double)success / images.GetLength(0), 2) * 100);
                file.WriteLine("False Positives:\t\t {0}\t{1}%", falsePositives, Math.Round((double)falsePositives / images.GetLength(0), 2) * 100);
                file.WriteLine("False Negatives:\t\t {0}\t{1}%", falseNegatives, Math.Round((double)falseNegatives / images.GetLength(0), 2) * 100);
                file.WriteLine();
            }

            Console.WriteLine("Detections:\t\t\t {0}\t{1}%", positives, Math.Round((double)positives / results.ToList().Count(r => r == true), 2) * 100);
            Console.WriteLine("Successful detections:\t\t {0}\t{1}%", success, Math.Round((double)success / images.GetLength(0), 2) * 100);
            Console.WriteLine("False Positives:\t\t {0}\t{1}%", falsePositives, Math.Round((double)falsePositives / images.GetLength(0), 2) * 100);
            Console.WriteLine("False Negatives:\t\t {0}\t{1}%", falseNegatives, Math.Round((double)falseNegatives / images.GetLength(0), 2) * 100);
        }

        public double[][,] IntegrateImages(double[][,] images)
        {
            var integralImages = new double[images.GetLength(0)][,];

            for (int i = 0; i < images.GetLength(0); i++)
            {
                integralImages[i] = IntegrateImage(images[i]);
            }

            return integralImages;
        }

        public double[,] IntegrateImage(double[,] image)
        {
            var _height = image.GetLength(0);
            var _width = image.GetLength(1);

            var integralImage = new double[_height + 1, _width + 1];

            integralImage[1, 1] = image[0, 0]; // Init 0, 0 value

            // Integrate first column
            for (int i = 1; i < _height + 1; i++)
            {
                integralImage[i, 1] = integralImage[i - 1, 1] + image[i - 1, 0];
            }

            // Integrate first row
            for (int i = 1; i < _width + 1; i++)
            {
                integralImage[1, i] = integralImage[1, i - 1] + image[0, i - 1];
            }

            // Integrate inner rectangle
            for (int i = 2; i < _height + 1; i++)
            {
                for (int j = 2; j < _width + 1; j++)
                {
                    integralImage[i, j] = integralImage[i - 1, j] + integralImage[i, j - 1] - integralImage[i - 1, j - 1] + image[i - 1, j - 1];
                }
            }

            return integralImage;
        }
    }
}
