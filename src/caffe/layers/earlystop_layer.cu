#include <algorithm>
#include <vector>
#include <limits>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        Dtype tmp = 0;
        Dtype val_loss_mean = 0;
        top[0]->mutable_cpu_data()[0] = bottom_data[0];
        if (Caffe::phase() == Caffe::TRAIN) {
            if (train_loss.size() > time_interval_ && index % iter_train_ == 1) {
                LOG(INFO) << "Start CHECKING" << std::endl;
                std::ifstream tmp2(path_tmp_.c_str(), std::ios::in | std::ios::binary);
                tmp2.read((char *) &val_list, sizeof(val_list));
                tmp2.close();
                //LOG(INFO) << "Read min_val " << val_list[0] << " from tmp.bin" << std::endl;
                //LOG(INFO) << "Read current_val " << val_list[1] << " from tmp.bin" << std::endl;
                minimum = EarlystopLayer<Dtype>::find_min(train_loss, time_interval_);
                //median = EarlystopLayer<Dtype>::find_median(train_loss, time_interval_);
                sum_loss = EarlystopLayer<Dtype>::sum_lastk(train_loss, time_interval_);
                //LOG(INFO) << "MINIMUM is " << minimum << std::endl;
                //LOG(INFO) << "MEDIAN is " << median << std::endl;
                //LOG(INFO) << "SUM is " << sum_loss << std::endl;
                tmp = sum_loss / (time_interval_ * minimum);
                tmp = (val_list[1] / val_list[0] - 1) / tmp / lamina_;
                LOG(INFO) << "The value for comparison is " << tmp << std::endl;
                if (tmp > threshold_) {
                    stop = true;
                }
                
                if (stop == true) {
                    bottom[0]->mutable_cpu_diff()[0] = 1.5;
                    LOG(INFO) << "Sub-task should be terminated" << std::endl;
                } else {
                    LOG(INFO) << "Sub-task should continue" << std::endl;
                }
            }
            if (index % iter_train_ == 0) {
                train_loss.push_back(bottom_data[0]);
            }
            index = index + 1;
            //LOG(INFO) << "index is " << index << std::endl;
            //LOG(INFO) << "Size of train_loss is " << train_loss.size() << std::endl;
            //LOG(INFO) << "Passed Loss Value is " << bottom_data[0] << std::endl;
            //LOG(INFO) << "EARLYSTOP Training phase, latest train loss is " << train_loss.back() << std::endl;
            
        } else if (Caffe::phase() == Caffe::TEST) {
            //LOG(INFO) << "EARLYSTOP Testing phase" << std::endl;
            val_loss.push_back(bottom_data[0]);
            if (val_loss.size() > 0 && val_loss.size() % iter_test_ == 0) {
                val_loss_mean = EarlystopLayer<Dtype>::sum_lastk(val_loss, iter_test_) / iter_test_;
                if (val_list[0] == 0) {
                    val_list[0] = val_loss_mean;
                    val_list[1] = val_loss_mean;
                    std::ofstream tmp1(path_tmp_.c_str(), std::ios::out | std::ios::binary);
                    tmp1.write((char *) &val_list, sizeof(val_list));
                    tmp1.close();
                } else {
                    if (val_list[0] > val_loss_mean) {
                        val_list[0] = val_loss_mean;
                    }
                    val_list[1] = val_loss_mean;
                    std::ofstream tmp1(path_tmp_.c_str(), std::ios::out | std::ios::binary);
                    tmp1.write((char *) &val_list, sizeof(val_list));
                    tmp1.close();
                }
                
                //LOG(INFO) << "Wrote min_val " << val_list[0] << " to tmp.bin" << std::endl;
                //LOG(INFO) << "Wrote current_val " << val_list[1] << " to tmp.bin" << std::endl;
            }
            
            //LOG(INFO) << "iter_test is " << iter_test << std::endl;
        }
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom) {
        LOG(INFO) << "Enter Backward propagation of Earlystop Layer" << std::endl;
        if (stop == true) {
            bottom[0]->mutable_cpu_diff()[0] = 0;
            LOG(INFO) << "EarlyStop Gradient is 0, update will be terminated" << std::endl;
        } else {
            bottom[0]->mutable_cpu_diff()[0] = 1;
            LOG(INFO) << "EarlyStop Gradient is 1, update will continue" << std::endl;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(EarlystopLayer);
} // namespace caffe
