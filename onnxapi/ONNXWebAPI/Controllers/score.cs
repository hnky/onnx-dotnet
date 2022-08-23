using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ONNXWebAPI.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Text;

namespace ONNXWebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ScoreController : ControllerBase
    {
        private readonly ILogger<ScoreController> _logger;

        public ScoreController(ILogger<ScoreController> logger)
        {
            _logger = logger;
        }

        [HttpGet(Name = "Score")]
        public async Task<Prediction> PredictAsync(string url)
        {
           // var imageFilePath = "Models/Homer.jpg";
            var modelFilePath = "ONNXModels/model.onnx";
            var labelFilePath = "ONNXModels/labels.txt";

            string[] lines;
            var list = new List<string>();
            var fileStream = new FileStream(labelFilePath, FileMode.Open, FileAccess.Read);
            using (var streamReader = new StreamReader(fileStream, Encoding.UTF8))
            {
                string line;
                while ((line = streamReader.ReadLine()) != null)
                {
                    list.Add(line);
                }
            }
            lines = list.ToArray();


            using (HttpClient client = new HttpClient())
            {
                using (HttpResponseMessage response = await client.GetAsync(url))
                using (Stream streamToReadFrom = await response.Content.ReadAsStreamAsync())
                {
                    // Transform Image
                    using Image<Rgb24> image = Image.Load<Rgb24>(streamToReadFrom, out IImageFormat format);

                    using Stream imageStream = new MemoryStream();
                    image.Mutate(x =>
                    {
                        x.Resize(new ResizeOptions
                        {
                            Size = new Size(224, 224),
                            Mode = ResizeMode.Crop
                        });
                    });
                    image.Save(imageStream, format);

                    // Preprocess image
                    Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                    var mean = new[] { 0.485f, 0.456f, 0.406f };
                    var stddev = new[] { 0.229f, 0.224f, 0.225f };
                    image.ProcessPixelRows(accessor =>
                    {
                        for (int y = 0; y < accessor.Height; y++)
                        {
                            Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                            for (int x = 0; x < accessor.Width; x++)
                            {
                                input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                                input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                                input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                            }
                        }
                    });

                    // Setup inputs
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input.1", input)
                    };

                    // Run inference
                    using var session = new InferenceSession(modelFilePath);
                    using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

                    // Postprocess to get softmax vector
                    IEnumerable<float> output = results.First().AsEnumerable<float>();
                    float sum = output.Sum(x => (float)Math.Exp(x));
                    IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

                    // Extract top 10 predicted classes
                    IEnumerable<Prediction> top10 = softmax.Select((x, i) => new Prediction { Label = lines[i], Confidence = x })
                                       .OrderByDescending(x => x.Confidence)
                                       .Take(10);

                    return top10.First();
                }
            }

           
        }
    }
}