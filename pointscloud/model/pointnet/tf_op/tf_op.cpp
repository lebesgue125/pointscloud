#include<cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


using namespace tensorflow;

REGISTER_OP("FarthestPointSample")
    .Input("input_tensor:float32")
    .Output("output_index: int32")
    .Output("output_points: float32")
    .Attr("points_num: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle dims;
        int points_num;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims));
        TF_RETURN_IF_ERROR(c->GetAttr("points_num", &points_num)); 
        c->set_output(0, c->MakeShape({c->Dim(dims, 0), points_num}));
        c->set_output(1, c->MakeShape({c->Dim(dims, 0), points_num, c->Dim(dims, 2)}));
        return Status::OK();
    });

REGISTER_OP("FarthestPointSampleGrad")
    .Input("input_index:int32")
    .Input("input_grad: float32")
    .Output("output_grad: float32")
    .Attr("batch_size: int")
    .Attr("points_num: int")
    .Attr("out_points_num: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle dims;
        int batchSize, pointsNum, outPointsNum;
        TF_RETURN_IF_ERROR(c->GetAttr("batch_size", &batchSize));
        TF_RETURN_IF_ERROR(c->GetAttr("points_num", &pointsNum));
        TF_RETURN_IF_ERROR(c->GetAttr("out_points_num", &outPointsNum));        
        c->set_output(0, c->MakeShape({batchSize, pointsNum, 3}));
        return Status::OK();
    });    

void farthestPointSamplingLauncher(int batchSize, int nPoints, int outNPoints, 
    const float * dataset, float * temp, int * indexs, float * outPoints);

class FarthestPointSampleOp: public OpKernel{
  public:
    explicit FarthestPointSampleOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("points_num", &out_n_points));
                    OP_REQUIRES(context, out_n_points > 0, errors::InvalidArgument("FarthestPointSample expects positive points number"));
                }
    void Compute(OpKernelContext * context) override {

      const Tensor& input_tensor = context->input(0);
      int batch_size = input_tensor.shape().dim_size(0);
      int n_points = input_tensor.shape().dim_size(1);
      OP_REQUIRES(context, input_tensor.dims() == 3 && input_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      auto input_flat = input_tensor.flat<float>();
      const float * input = &(input_flat(0));
      
      Tensor * out_tensor_index, * out_tensor_points;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{batch_size, out_n_points}, &out_tensor_index));
      auto out_flat_index = out_tensor_index->flat<int>();
      int * out_index = &(out_flat_index(0));

      OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{batch_size, out_n_points, 3}, &out_tensor_points));
      auto out_flat_points = out_tensor_points->flat<float>();
      float * out_points = &(out_flat_points(0));

      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{batch_size, n_points},&temp_tensor));
      auto temp_flat = temp_tensor.flat<float>();
      float * temp = &(temp_flat(0));
      farthestPointSamplingLauncher(batch_size, n_points, out_n_points, input, temp, out_index, out_points);
    }

    private:
        int out_n_points;
};

REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleOp);

void farthestPointSampleGrad(int batchSize, int nPoints, int outNPoints, const int * pointsIndexs, const float * pointsGrad, float * backGrad);
class FarthestPointSampleGradOp: public OpKernel{
  private: 
    int batchSize, pointsNum, outPointsNum;
  public:
    explicit FarthestPointSampleGradOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize));
                    OP_REQUIRES_OK(context, context->GetAttr("points_num", &pointsNum));
                    OP_REQUIRES_OK(context, context->GetAttr("outPointsNum", &outPointsNum));}
    
    void Compute(OpKernelContext * context) override {
      const Tensor & inputIndexTensor = context->input(0);
      const int * inputIndex = &(inputIndexTensor.flat<int>()(0));
      
      const Tensor & inputGradTensor = context->input(1);
      const float * inputGrad = &(inputGradTensor.flat<float>()(0));
      
      Tensor * backGradTensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{batchSize,pointsNum ,3},&backGradTensor));
      float * backGrad = &(backGradTensor->flat<float>()(0));

      farthestPointSampleGrad(batchSize, pointsNum, outPointsNum, inputIndex, inputGrad, backGrad);
    }
};