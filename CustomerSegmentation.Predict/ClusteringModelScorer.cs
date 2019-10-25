using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using Common;
using CustomerSegmentation.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations;

namespace CustomerSegmentation.Model
{
    public class ClusteringModelScorer
    {
        private readonly string _pivotDataLocation;

        private readonly string _plotLocation;
        private readonly string _csvlocation;
        private readonly MLContext _mlContext;
        private ITransformer _trainedModel;

        public ClusteringModelScorer(MLContext mlContext, string pivotDataLocation, string plotLocation, string csvlocation)
        {
            _pivotDataLocation = pivotDataLocation;
            _plotLocation = plotLocation;
            _csvlocation = csvlocation;
            _mlContext = mlContext;
        }

        public ITransformer LoadModel(string modelPath)
        {
            _trainedModel = _mlContext.Model.Load(modelPath, out var modelInputSchema);
            return _trainedModel;
        }

        public void CreateCustomerClusters()
        {
            var data = _mlContext.Data.LoadFromTextFile(path:_pivotDataLocation,
                            columns: new[]
                                        {
                                          new TextLoader.Column("Features", DataKind.Single, new[] {new TextLoader.Range(0, 31) }),
                                          new TextLoader.Column(nameof(PivotData.LastName), DataKind.String, 32)
                                        },
                            hasHeader: true,
                            separatorChar: ',');
            
            //Apply data transformation to create predictions/clustering
            var tranfomedDataView = _trainedModel.Transform(data);
            var predictions = _mlContext.Data.CreateEnumerable <ClusteringPrediction>(tranfomedDataView, false)
                            .ToArray();

            //Generate data files with customer data grouped by clusters
            SaveCustomerSegmentationCSV(predictions, _csvlocation);

            //Plot/paint the clusters in a chart and open it with the by-default image-tool in Windows
            SaveCustomerSegmentationPlotChart(predictions, _plotLocation);
            OpenChartInDefaultWindow(_plotLocation);
        }

        public void PredictCustomerClusters()
        {
            //var data = _mlContext.Data.LoadFromTextFile(path: _pivotDataLocation,
            //                columns: new[]
            //                            {
            //                              new TextLoader.Column("Features", DataKind.Single, new[] {new TextLoader.Range(0, 31) }),
            //                              new TextLoader.Column(nameof(PivotData.LastName), DataKind.String, 32)
            //                            },
            //                hasHeader: true,
            //                separatorChar: ',');


            var predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ClusteringPrediction>(_trainedModel);

            //Load sample data for prediction
            var data = new ModelInput
            {
                Features = new float[] { 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                LastName = "Mera"
            };

            // Try model on sample data
            ClusteringPrediction result = predEngine.Predict(data);
            List<ClusteringPrediction> enumResult = new List<ClusteringPrediction>
            {
                result
            };

            Console.WriteLine($"\nCustomer: {data.LastName} | Prediction: {result.SelectedClusterId} cluster");
            Console.WriteLine($"Location X: {result.Location[0].ToString()} | Location Y: {result.Location[1].ToString()} ");

            SaveCustomerSegmentationPlotChartSingle(enumResult.ToArray(), "customerSegmentation2.svg");
            OpenChartInDefaultWindow("customerSegmentation2.svg");
        }

        private class ModelInput
        {
            [LoadColumn(0), VectorType(32), ColumnName("Features")]
            public float[] Features = new float[31];

            [LoadColumn(1), ColumnName("LastName")]
            public string LastName;
        }

        private static void SaveCustomerSegmentationCSV(IEnumerable<ClusteringPrediction> predictions, string csvlocation)
        {
            ConsoleHelper.ConsoleWriteHeader("CSV Customer Segmentation");
            using (var w = new System.IO.StreamWriter(csvlocation))
            {
                w.WriteLine($"LastName,SelectedClusterId");
                w.Flush();
                predictions.ToList().ForEach(prediction => {
                    w.WriteLine($"{prediction.LastName},{prediction.SelectedClusterId}");
                    w.Flush();
                });
            }

            Console.WriteLine($"CSV location: {csvlocation}");
        }

        private static void SaveCustomerSegmentationPlotChart(IEnumerable<ClusteringPrediction> predictions, string plotLocation)
        {
            Common.ConsoleHelper.ConsoleWriteHeader("Plot Customer Segmentation");

            var plot = new PlotModel { Title = "Customer Segmentation", IsLegendVisible = true };

            var clusters = predictions.Select(p => p.SelectedClusterId).Distinct().OrderBy(x => x);

            foreach (var cluster in clusters)
            {
                var scatter = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerStrokeThickness = 2, Title = $"Cluster: {cluster}", RenderInLegend=true };
                var series = predictions
                    .Where(p => p.SelectedClusterId == cluster)
                    .Select(p => new ScatterPoint(p.Location[0], p.Location[1])).ToArray();
                scatter.Points.AddRange(series);
                plot.Series.Add(scatter);
            }

            plot.DefaultColors = OxyPalettes.HueDistinct(plot.Series.Count).Colors;

            var exporter = new SvgExporter { Width = 600, Height = 400 };
            using (var fs = new System.IO.FileStream(plotLocation, System.IO.FileMode.Create))
            {
                exporter.Export(plot, fs);
            }

            Console.WriteLine($"Plot location: {plotLocation}");
        }

        private static void SaveCustomerSegmentationPlotChartSingle(IEnumerable<ClusteringPrediction> predictions, string plotLocation)
        {
            Common.ConsoleHelper.ConsoleWriteHeader("Plot Customer Segmentation");

            var plot = new PlotModel { Title = "Customer Segmentation", IsLegendVisible = true };

            var clusters = predictions.Select(p => p.SelectedClusterId).Distinct().OrderBy(x => x);

            foreach (var cluster in clusters)
            {
                var scatter = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerStrokeThickness = 2, Title = $"Cluster: {cluster}", RenderInLegend = true };
                var series = predictions
                    .Where(p => p.SelectedClusterId == cluster)
                    .Select(p => new ScatterPoint(p.Location[0], p.Location[1])).ToArray();
                scatter.Points.AddRange(series);
                plot.Series.Add(scatter);
            }

            plot.DefaultColors = OxyPalettes.HueDistinct(10).Colors;

            var exporter = new SvgExporter { Width = 600, Height = 400 };
            using (var fs = new System.IO.FileStream(plotLocation, System.IO.FileMode.Create))
            {
                exporter.Export(plot, fs);
            }

            Console.WriteLine($"Plot location: {plotLocation}");
        }

        private static void OpenChartInDefaultWindow(string plotLocation)
        {
            Console.WriteLine("Showing chart...");
            var p = new Process();
            p.StartInfo = new ProcessStartInfo(plotLocation)
            {
                UseShellExecute = true
            };
            p.Start();
        }
    }
}
