#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>

namespace caffe2 {

void print(const Blob *blob, const std::string &name) {
  auto tensor = blob->Get<TensorCPU>();
  const auto &data = tensor.data<float>();
  std::cout << name << "(" << tensor.dims()
            << "): [" << std::vector<float>(data, data + tensor.size()) << ']'
            << std::endl;
}

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Toy Regression Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-toy-regression.html"
            << std::endl;
  std::cout << std::endl;


  // # Tutorial 2. A Simple Toy Regression
  //
  // This is a quick example showing how one can use the concepts introduced in
  // Tutorial 1 (Basics) to do a quick toy regression.
  //
  // The problem we are dealing with is a very simple one, with two-dimensional
  // input `x` and one-dimensional output `y`, and a weight vector `w=[2.0,
  // 1.5]` and bias `b=0.5`. The equation to generate ground truth is:
  //
  // ```y = wx + b```
  //
  // For this tutorial, we will be generating training data using Caffe2
  // operators as well. Note that this is usually not the case in your daily
  // training jobs: in a real training scenario data is usually loaded from an
  // external source, such as a Caffe DB (i.e. a key-value storage) or a Hive
  // table. We will cover this in the MNIST tutorial.
  //
  // We will write out every piece of math in Caffe2 operators. This is often an
  // overkill if your algorithm is relatively standard, such as CNN models. In
  // the MNIST tutorial, we will show how to use the CNN model helper to more
  // easily construct CNN models.

  // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize
  Workspace workspace;


  // ## Declaring the computation graphs
  //
  // There are two graphs that we declare: one is used to initialize the various
  // parameters and constants that we are going to use in the computation, and
  // another main graph that is used to run stochastic gradient descent.
  //
  // First, the init net: note that the name does not matter, we basically want
  // to put the initialization code in one net so we can then call RunNetOnce()
  // to execute it. The reason we have a separate init_net is that, these
  // operators do not need to run more than once for the whole training
  // procedure.

  // >>> init_net = core.Net("init")
  NetDef initModel;
  initModel.set_name("init");

  // >>> W_gt = init_net.GivenTensorFill([], "W_gt", shape=[1, 2],
  //                                     values=[2.0, 1.5])
  {
    // NetUtil init_model_util(initModel);
    // init_model_util.AddGivenTensorFillOp(TensorCPU({1, 2}, {2.0, 5.0},
    //                                                nullptr), "W_gt");
    auto op = initModel.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    arg1->add_ints(2);
    auto arg2 = op->add_arg();
    arg2->set_name("values");
    arg2->add_floats(2.0);
    arg2->add_floats(1.5);
    op->add_output("W_gt");
  }

  // >>> B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
  {
    auto op = initModel.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("values");
    arg2->add_floats(0.5);
    op->add_output("B_gt");
  }

  // >>> ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_f(1.0);
    op->add_output("ONE");
  }

  // >>> ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
  // dtype=core.DataType.INT32)
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_i(0);
    auto arg3 = op->add_arg();
    arg3->set_name("dtype");
    arg3->set_i(TensorProto_DataType_INT32);
    op->add_output("ITER");
  }

  // >>> W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
  {
    auto op = initModel.add_op();
    op->set_type("UniformFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    arg1->add_ints(2);
    auto arg2 = op->add_arg();
    arg2->set_name("min");
    arg2->set_f(-1);
    auto arg3 = op->add_arg();
    arg3->set_name("max");
    arg3->set_f(1);
    op->add_output("W");
  }

  // >>> B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_f(0);
    op->add_output("B");
  }

  std::cout << initModel.DebugString() << "--------" << std::endl;

  // >>> train_net = core.Net("train")
  NetDef trainModel;
  trainModel.set_name("train");

  // >>> X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0,
  //                                run_once=0)
  {
    auto op = trainModel.add_op();
    op->set_type("GaussianFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(64);
    arg1->add_ints(2);
    auto arg2 = op->add_arg();
    arg2->set_name("mean");
    arg2->set_f(0);
    auto arg3 = op->add_arg();
    arg3->set_name("std");
    arg3->set_f(1);
    auto arg4 = op->add_arg();
    arg4->set_name("run_once");
    arg4->set_i(0);
    op->add_output("X");
  }

  // >>> Y_gt = X.FC([W_gt, B_gt], "Y_gt")
  {
    auto op = trainModel.add_op();
    op->set_type("FC");
    op->add_input("X");
    op->add_input("W_gt");
    op->add_input("B_gt");
    op->add_output("Y_gt");
  }

  // >>> noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0,
  // std=1.0, run_once=0)
  {
    auto op = trainModel.add_op();
    op->set_type("GaussianFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(64);
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("mean");
    arg2->set_f(0);
    auto arg3 = op->add_arg();
    arg3->set_name("std");
    arg3->set_f(1);
    auto arg4 = op->add_arg();
    arg4->set_name("run_once");
    arg4->set_i(0);
    op->add_output("noise");
  }

  // >>> Y_noise = Y_gt.Add(noise, "Y_noise")
  {
    auto op = trainModel.add_op();
    op->set_type("Add");
    op->add_input("Y_gt");
    op->add_input("noise");
    op->add_output("Y_noise");
  }

  // >>> Y_noise = Y_noise.StopGradient([], "Y_noise")
  {
    auto op = trainModel.add_op();
    op->set_type("StopGradient");
    op->add_input("Y_noise");
    op->add_output("Y_noise");
  }

  std::vector<OperatorDef *> gradient_ops;

  // >>> Y_pred = X.FC([W, B], "Y_pred")
  {
    auto op = trainModel.add_op();
    op->set_type("FC");
    op->add_input("X");
    op->add_input("W");
    op->add_input("B");
    op->add_output("Y_pred");
    gradient_ops.push_back(op);
  }

  // >>> dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
  {
    auto op = trainModel.add_op();
    op->set_type("SquaredL2Distance");
    op->add_input("Y_noise");
    op->add_input("Y_pred");
    op->add_output("dist");
    gradient_ops.push_back(op);
  }

  // >>> loss = dist.AveragedLoss([], ["loss"])
  {
    auto op = trainModel.add_op();
    op->set_type("AveragedLoss");
    op->add_input("dist");
    op->add_output("loss");
    gradient_ops.push_back(op);
  }

  // >>> gradient_map = train_net.AddGradientOperators([loss])
  {
    auto op = trainModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("value");
    arg->set_f(1.0);
    op->add_input("loss");
    op->add_output("loss_grad");
    op->set_is_gradient_op(true);
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  for (auto op : gradient_ops) {
    vector<GradientWrapper> output(op->output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op->output(i) + "_grad";
    }
    GradientOpsMeta meta = GetGradientForOp(*op, output);
    auto grad = trainModel.add_op();
    grad->CopyFrom(meta.ops_[0]);
    grad->set_is_gradient_op(true);
  }

  // >>> train_net.Iter(ITER, ITER)
  {
    auto op = trainModel.add_op();
    op->set_type("Iter");
    op->add_input("ITER");
    op->add_output("ITER");
  }

  // >>> LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
  //                                 stepsize=20, gamma=0.9)
  {
    auto op = trainModel.add_op();
    op->set_type("LearningRate");
    auto arg1 = op->add_arg();
    arg1->set_name("base_lr");
    arg1->set_f(-0.1);
    auto arg2 = op->add_arg();
    arg2->set_name("policy");
    arg2->set_s("step");
    auto arg3 = op->add_arg();
    arg3->set_name("stepsize");
    arg3->set_i(20);
    auto arg4 = op->add_arg();
    arg4->set_name("gamma");
    arg4->set_f(0.9);
    op->add_input("ITER");
    op->add_output("LR");
  }

  // >>> train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
  {
    auto op = trainModel.add_op();
    op->set_type("WeightedSum");
    op->add_input("W");
    op->add_input("ONE");
    op->add_input("W_grad");
    op->add_input("LR");
    op->add_output("W");
  }

  // >>> train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
  {
    auto op = trainModel.add_op();
    op->set_type("WeightedSum");
    op->add_input("B");
    op->add_input("ONE");
    op->add_input("B_grad");
    op->add_input("LR");
    op->add_output("B");
  }

  // print(trainModel);
  std::cout << trainModel.DebugString() << "--------" << std::endl;

  // >>> workspace.RunNetOnce(init_net)
  // >>> workspace.CreateNet(train_net)
  CAFFE_ENFORCE(workspace.RunNetOnce(initModel));
  CAFFE_ENFORCE(workspace.CreateNet(trainModel));

  // >>> print("Before training, W is: {}".format(workspace.FetchBlob("W")))
  // >>> print("Before training, B is: {}".format(workspace.FetchBlob("B")))
  print(workspace.GetBlob("W"), "Before, W");
  print(workspace.GetBlob("B"), "Before, B");

  // >>> for i in range(100):
  //         workspace.RunNet(train_net.Proto().name)
  for (auto i = 1; i <= 10000; i++) {
    CAFFE_ENFORCE(workspace.RunNet(trainModel.name()));
    if ((i + 1) % 1000 == 0 || i == 1) {
      float w = workspace.GetBlob("W")->Get<TensorCPU>().data<float>()[0];
      float b = workspace.GetBlob("B")->Get<TensorCPU>().data<float>()[0];
      float loss = workspace.GetBlob("loss")->Get<TensorCPU>().data<float>()[0];
      std::cout << "    step: " << std::setw(4) << i
                << ", W: " << std::setw(9) << w
                << " B: " << b << " loss: " << loss << std::endl;
    }
  }

  // >>> print("After training, W is: {}".format(workspace.FetchBlob("W")))
  // >>> print("After training, B is: {}".format(workspace.FetchBlob("B")))
  print(workspace.GetBlob("W"), "After, W");
  print(workspace.GetBlob("B"), "After, B");

  // >>> print("Ground truth W is: {}".format(workspace.FetchBlob("W_gt")))
  // >>> print("Ground truth B is: {}".format(workspace.FetchBlob("B_gt")))
  print(workspace.GetBlob("W_gt"), "ground truth W");
  print(workspace.GetBlob("B_gt"), "ground truth B");
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
