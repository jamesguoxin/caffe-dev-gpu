#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  // James added code
  if (top[0]->count() == 1 && top[1]->count() == 1 && top[1]->cpu_diff()[0] == 1.5) {
    top[0]->mutable_cpu_diff()[0] = 0;
    top[1]->mutable_cpu_diff()[0] = 0;
  }
  //LOG(INFO) << "split 0 gpu_diff is " << top[0]->cpu_diff()[0] <<std::endl;
  //LOG(INFO) << "split 1 gpu_diff is " << top[1]->cpu_diff()[0] <<std::endl;
  // James added code
  //////////////////////////////////////////////////////////////////////////////
  caffe_gpu_add(count_, top[0]->gpu_diff(), top[1]->gpu_diff(),
                bottom[0]->mutable_gpu_diff());
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
