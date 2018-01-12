#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

CAFFE2_DEFINE_string(init_net, "res/squeezenet_init_net.pb",
                     "The given path to the init protobuffer.");
CAFFE2_DEFINE_string(predict_net, "res/squeezenet_predict_net.pb",
                     "The given path to the predict protobuffer.");
CAFFE2_DEFINE_string(file, "res/image_file.jpg", "The image file.");
CAFFE2_DEFINE_string(classes, "res/imagenet_classes.txt", "The classes file.");
CAFFE2_DEFINE_int(size, 227, "Image size (square) to pass to model.");

namespace caffe2 {

void run() {
  std::cout << "\n## Caffe2 Loading Pre-Trained Models Tutorial ##\n";
  std::cout << "https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html";
  std::cout << "\nhttps://caffe2.ai/docs/tutorial-image-pre-processing.html\n"
            << std::endl;

  if (!std::ifstream(FLAGS_init_net) || !std::ifstream(FLAGS_predict_net)) {
    std::cerr << "error: Squeezenet model file missing: "
              << (std::ifstream(FLAGS_init_net) ? FLAGS_predict_net
                                                       : FLAGS_init_net);
    std::cerr << "\nMake sure to first run ./script/download_resource.sh\n";
    return;
  }
  if (!std::ifstream(FLAGS_file).good()) {
    std::cerr << "error: Image file missing: " << FLAGS_file << std::endl;
    return;
  }
  if (!std::ifstream(FLAGS_classes).good()) {
    std::cerr << "error: Classes file invalid: " << FLAGS_classes << std::endl;
    return;
  }

  std::cout << "init-net: " << FLAGS_init_net << std::endl;
  std::cout << "predict-net: " << FLAGS_predict_net << std::endl;
  std::cout << "file: " << FLAGS_file << std::endl;
  std::cout << "size: " << FLAGS_size << std::endl << std::endl;

  // >>> img =
  // skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
  auto image = cv::imread(FLAGS_file);  // CV_8UC3
  std::cout << "image size: " << image.size() << std::endl;

  // scale image to fit
  cv::Size scale(std::max(FLAGS_size * image.cols / image.rows, FLAGS_size),
                 std::max(FLAGS_size, FLAGS_size * image.rows / image.cols));
  cv::resize(image, image, scale);
  std::cout << "scaled size: " << image.size() << std::endl;

  // crop image to fit
  cv::Rect crop((image.cols - FLAGS_size) / 2, (image.rows - FLAGS_size) / 2,
                FLAGS_size, FLAGS_size);
  image = image(crop);
  std::cout << "cropped size: " << image.size() << std::endl;

  // convert to float, normalize to mean 128
  image.convertTo(image, CV_32FC3, 1.0, -128);
  std::cout << "value range: ("
            << *std::min_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ", "
            << *std::max_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  std::vector<float> data;
  for (auto &c : channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }
  std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});
  TensorCPU input_as_tensor(dims, data, NULL);

  // Load Squeezenet model
  NetDef init_net, predict_net;

  // >>> with open(path_to_INIT_NET) as f: init_net = f.read()
  // >>> with open(path_to_PREDICT_NET) as f: predict_net = f.read()
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

  // >>> p = workspace.Predictor(init_net, predict_net)
  Predictor predictor(init_net, predict_net);

  // >>> results = p.run([img])
  Predictor::TensorVector inputVec({&input_as_tensor}), outputVec;
  predictor.run(inputVec, &outputVec);
  const TensorCPU& output = *(outputVec[0]);

  // sort top results
  const auto &probs = output.data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < output.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }

  std::sort(pairs.begin(), pairs.end());

  std::cout << std::endl;

  // read classes
  std::ifstream file(FLAGS_classes);
  std::string temp;
  std::vector<std::string> classes;
  while (std::getline(file, temp)) {
    classes.push_back(temp);
  }

  // show results
  std::cout << "output: " << std::endl;
  for (auto pair : pairs) {
    std::cout << "  " << std::setw(2) << pair.first << "% "
              << std::setw(14) << "'" + classes[pair.second] + "'" << " ("
              << std::setw(3) << pair.second << ")" << std::endl;
  }
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
