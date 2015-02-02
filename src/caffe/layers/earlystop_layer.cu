#include <algorithm>
#include <vector>
#include <limits>
#include <fstream>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype tmp = 0;
        Dtype val_loss_mean = 0;
        top_data[0] = bottom_data[0];
        if (Caffe::phase() == Caffe::TRAIN) {
            if (index / iter_train_ > time_interval_ && index % iter_train_ == 1) {         // Only start to check after time_interval_ epochs
                LOG(INFO) << "Start CHECKING" << std::endl;
                std::ifstream tmp2(path_tmp_.c_str(), std::ios::in | std::ios::binary);
                tmp2.read(reinterpret_cast<char*>(&val_list[0]), val_list.size()*sizeof(Dtype));
                tmp2.close();
                minimum = EarlystopLayer<Dtype>::find_min(val_list, time_interval_);
                //sum_loss = EarlystopLayer<Dtype>::sum_lastk(val_list, time_interval_);
                LOG(INFO) << "MINIMUM is " << minimum << std::endl;
                for (int idx = 0; idx < val_list.size() - 1; idx++) {
                    if ((val_list[idx] - val_list[idx+1]) > 0) {
                        tmp += fabs(val_list[idx] - val_list[idx+1]) / val_list[idx];
                    }
                }
                tmp = tmp / (val_list.size() - 1);
                tmp = tmp * lamina_;
                LOG(INFO) << "The value for comparison is " << tmp << std::endl;
                if (tmp < threshold_) {
                    stop = true;
                }
                
                if (stop == true) {
                    bottom_diff[0] = 1.5;
                    LOG(INFO) << "Sub-task should be terminated" << std::endl;
                } else {
                    LOG(INFO) << "Sub-task should continue" << std::endl;
                }
            }
            if (index % iter_train_ == 0) {
                train_loss.push_back(bottom_data[0]);
            }
            index = index + 1;
            
        } else if (Caffe::phase() == Caffe::TEST) {
            //LOG(INFO) << "EARLYSTOP Testing phase" << std::endl;
            val_loss.push_back(bottom_data[0]);
            if (val_loss.size() > 0 && val_loss.size() % iter_test_ == 0) {
                val_loss_mean = EarlystopLayer<Dtype>::sum_lastk(val_loss, iter_test_) / iter_test_;
                if (val_list.size() > (val_loss.size() / iter_test_ - 1)) {
                    val_list[val_loss.size() / iter_test_ - 1] = val_loss_mean;
                } else {
                    for (int idx = 0; idx < val_list.size() - 1; idx ++) {
                        val_list[idx] = val_list[idx + 1];
                    }
                    val_list[val_list.size() - 1] = val_loss_mean;
                }
                //LOG(INFO) << "Last value of Val_List is " << val_list[val_list.size() - 1] << std::endl;
                //LOG(INFO) << "Val Loss is " << val_loss_mean << std::endl;
                std::ofstream tmp1(path_tmp_.c_str(), std::ios::out | std::ios::binary);
                tmp1.write(reinterpret_cast<char*>(&val_list[0]), val_list.size()*sizeof(Dtype));
                tmp1.close();
            }
        }
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom) {
        LOG(INFO) << "Enter Backward propagation of Earlystop Layer" << std::endl;
        Dtype* bottom_data = bottom[0]->mutable_cpu_diff();
        if (stop == true) {
            bottom_data[0] = 0;
            LOG(INFO) << "EarlyStop Gradient is 0, update will be terminated" << std::endl;
        } else {
            bottom_data[0] = 1;
            LOG(INFO) << "EarlyStop Gradient is 1, update will continue" << std::endl;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(EarlystopLayer);
} // namespace caffe
