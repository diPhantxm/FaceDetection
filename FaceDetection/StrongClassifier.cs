using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Drawing;
using System.IO;

namespace FaceDetection
{
    [Serializable]
    public class StrongClassifier : BaseClassifier
    {
        public List<WeakClassifier> WeakClassifiers { get; private set; }
        private List<double> Betas;
        private List<double> Weights;
        private List<double[]> ImagesWeights;

        public StrongClassifier()
        {

        }

        public StrongClassifier(IEnumerable<WeakClassifier> classifiers, double[] weights)
        {
            WeakClassifiers = classifiers.ToList();
            Weights = weights.ToList();
            Parity = -1;
        }

        public double[][,] IntegrateImages(double[][,] images)
        {
            var integralImages = new double[images.GetLength(0)][,];

            for (int i = 0; i < images.GetLength(0); i++)
            {
                integralImages[i] = new double[images[i].GetLength(0) + 1, images[i].GetLength(1) + 1];
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

        public void Train(double[][,] images, List<WeakClassifier> weakClassifiers, bool[] results, int featureCount)
        {
            if (results.Count(r => r) == 0) return;
            if (results.Count(r => !r) == 0) return;

            var integralImages = IntegrateImages(images);

            var negativeCount = results.ToList().Where(r => r == false).Count();
            var positiveCount = results.ToList().Where(r => r == true).Count();

            WeakClassifiers = new List<WeakClassifier>();

            Betas = new List<double>();
            Weights = new List<double>(); // Strong classifier Weights

            InitializeWeights(results);

            Console.WriteLine("Training started...");

            WeakClassifier minFeature = null;

            // Weak classifiers loop
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                // Normalize the weights
                var weightSum = ImagesWeights[featureIndex].Sum();
                for (int i = 0; i < images.Length; i++)
                {
                    ImagesWeights[featureIndex][i] /= weightSum;
                }

                // Train each weak classifier in parallel
                weakClassifiers.AsParallel().ForAll(wc =>
                {
                    wc.Train(ref integralImages, ref results, ImagesWeights[featureIndex]);
                });

                // Choosing weak classifier with lowest error
                minFeature = weakClassifiers.OrderBy(wc => wc.Error).First();

                // Saving weak classifier
                WeakClassifiers.Add(minFeature);
                weakClassifiers.Remove(minFeature);

                // Calc Betas and Weights
                Betas.Add(minFeature.Error / (1 - minFeature.Error));
                Weights.Add(Math.Log(1.0 / Betas[featureIndex]));

                // Weights update
                ImagesWeights.Add(new double[integralImages.Length]);
                for (int i = 0; i < integralImages.Length; i++)
                {
                    var featureResult = minFeature.Detect(integralImages[i]);

                    if (featureResult == results[i])
                    {
                        ImagesWeights[featureIndex + 1][i] = ImagesWeights[featureIndex][i] * Betas[featureIndex];
                    }
                    else
                    {
                        ImagesWeights[featureIndex + 1][i] = ImagesWeights[featureIndex][i];
                    }
                }

                Console.WriteLine("Feature #{0}. Error: {1}", featureIndex + 1, minFeature.Error);
            }

            var scores = CalcScores(integralImages, results);
            DetermineThreshold(ref scores);

            Statistics(images, results);

            Console.WriteLine("Training finished.");
        }

        public void AddWeakClassifier(double[][,] images, bool[] results, List<WeakClassifier> weakClassifiers)
        {
            if (results.Count(r => r) == 0) return;
            if (results.Count(r => !r) == 0) return;

            if (ImagesWeights == null) ImagesWeights = new List<double[]>();
            if (Weights == null) Weights = new List<double>();
            if (WeakClassifiers == null) WeakClassifiers = new List<WeakClassifier>();
            if (Betas == null) Betas = new List<double>();

            if (ImagesWeights.Count == 0) InitializeWeights(results);

            var integralImages = IntegrateImages(images);

            var negativeCount = results.ToList().Where(r => r == false).Count();
            var positiveCount = results.ToList().Where(r => r == true).Count();

            WeakClassifier minFeature = null;

            // Normalize the weights
            var weightSum = ImagesWeights.Last().Sum();
            for (int i = 0; i < images.Length; i++)
            {
                ImagesWeights.Last()[i] /= weightSum;
            }

            // Train each weak classifier in parallel
            weakClassifiers.AsParallel().ForAll(wc =>
            {
                wc.Train(ref integralImages, ref results, ImagesWeights.Last());
            });

            // Choosing weak classifier with lowest error
            minFeature = weakClassifiers.OrderBy(wc => wc.Error).First();

            if (double.IsNaN(minFeature.Error)) throw new Exception("Error is NaN");

            // Saving weak classifier
            WeakClassifiers.Add(minFeature);
            weakClassifiers.Remove(minFeature);

            // Calc Betas and Weights
            Betas.Add(minFeature.Error / (1 - minFeature.Error));
            Weights.Add(Math.Log(1.0 / Betas.Last()));

            // Weights update
            ImagesWeights.Add(new double[integralImages.Length]);
            for (int i = 0; i < integralImages.Length; i++)
            {
                var featureResult = minFeature.Detect(integralImages[i]);

                if (featureResult == results[i])
                {
                    ImagesWeights.Last()[i] = ImagesWeights[ImagesWeights.Count - 2][i] * Betas[ImagesWeights.Count - 2];
                }
                else
                {
                    ImagesWeights.Last()[i] = ImagesWeights[ImagesWeights.Count - 2][i];
                }
            }

            var scores = CalcScores(integralImages, results);
            DetermineThreshold(ref scores);

            var detectionRate = 0.0;
            var falsePositiveRate = 0.0;

            Evaluate(integralImages, results, out falsePositiveRate, out detectionRate);

            Console.WriteLine("Weak Classifier #{0} was added. Error: {1} Detection Rate: {2} False Positive Rate: {3}", 
                WeakClassifiers.Count, Math.Round(minFeature.Error, 5).ToString().PadRight(10), 
                Math.Round(detectionRate, 4).ToString().PadRight(10), 
                Math.Round(falsePositiveRate, 4).ToString().PadRight(10));
        }

        private List<Tuple<double, bool>> CalcScores(double[][,] integralImages, bool[] results)
        {
            var scores = new List<Tuple<double, bool>>();
            for (int i = 0; i < integralImages.Length; i++)
            {
                var score = 0.0;
                for (int j = 0; j < WeakClassifiers.Count; j++)
                {
                    score += WeakClassifiers[j].Detect(integralImages[i]) ? Weights[j] : 0;
                }

                scores.Add(new Tuple<double, bool>(score, results[i]));
            }

            return scores;
        }

        private void DetermineThreshold(ref List<Tuple<double, bool>> scores)
        {
            Threshold = scores.Where(s => s.Item2).Select(s => s.Item1).Min();
            //Threshold = Weights.Sum() / 2;
        }

        public void Evaluate(double[][,] images, bool[] results, out double fPositiveRate, out double detectionRate)
        {
            if (images.Length != results.Length) throw new ArgumentException("Images legnth does not equal to results length.");

            var falsePositives = 0;
            var detections = 0;

            for (int i = 0; i < images.Length; i++)
            {
                var result = DetectOnIntegral(images[i]);

                if (result) detections++;
                if (result && !results[i]) falsePositives++;
            }

            fPositiveRate = (double)falsePositives / results.Count(r => !r);
            detectionRate = (double)detections / images.Length;
        }

        public bool Detect(double[,] image)
        {
            var integralImage = IntegrateImage(image);

            return DetectOnIntegral(integralImage);
        }

        public bool DetectOnIntegral(double[,] integralImage)
        {
            var sum = .0;

            for (int i = 0; i < WeakClassifiers.Count; i++)
            {
                var featureResult = WeakClassifiers[i].Detect(integralImage);

                if (featureResult)
                {
                    sum += Weights[i];
                }
            }

            if (sum >= Threshold) return true;
            else return false;
        }

        public void DecreaseThreshold()
        {
            Threshold -= 0.001;
        }

        private void InitializeWeights(bool[] results)
        {
            var positiveCount = results.Count(r => r);
            var negativeCount = results.Count(r => !r);
            var total = positiveCount + negativeCount;

            ImagesWeights = new List<double[]>();
            ImagesWeights.Add(new double[total]);

            for (int i = 0; i < total; i++)
            {
                if (results[i])
                {
                    ImagesWeights[0][i] = 1.0 / (2 * positiveCount);
                }
                else
                {
                    ImagesWeights[0][i] = 1.0 / (2 * negativeCount);
                }
            }
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

            using (var file = new StreamWriter("B:\\Desktop\\Statisticss.txt", append: true))
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
    }
}