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
    Dtype EarlystopLayer<Dtype>::find_min(const std::vector<Dtype> loss_sequence,
                                          const size_t k) {
        Dtype result = 0;
        std::vector<Dtype> tmp(loss_sequence.end() - k, loss_sequence.end());
        std::sort(tmp.begin(), tmp.end());
        result = tmp[0];
        return result;
    }
    
    template <typename Dtype>
    Dtype EarlystopLayer<Dtype>::find_median(const std::vector<Dtype> loss_sequence,
                                             const size_t k) {
        Dtype result = 0;
        std::vector<Dtype> tmp(loss_sequence.end() - k, loss_sequence.end());
        std::sort(tmp.begin(), tmp.end());
        result = tmp[k / 2];
        return result;
    }
    
    template <typename Dtype>
    Dtype EarlystopLayer<Dtype>::sum_lastk(const std::vector<Dtype> loss_sequence,
                                           const size_t k) {
        Dtype result = 0;
        for (size_t i = loss_sequence.size() - k; i < loss_sequence.size(); i++) {
            result = result + loss_sequence[i];
        }
        return result;
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
        EarlystopParameter early_param = this->layer_param_.earlystop_param();
        threshold_ = early_param.threshold();
        lamina_ = early_param.lamina();
        time_interval_ = early_param.time_interval();          // This should be the number of epoches
        path_tmp_ = early_param.path_tmp();
        //LOG(INFO) << "path_tmp is " << path_tmp_ << std::endl;
        iter_test_ = early_param.iter_test();                  // Number of iterations per test
        iter_train_ = early_param.iter_train();                // Number of iterations per train epoch
        
        stop = false;
        minimum = 0;
        sum_loss = 0;
        index = 0;
        val_list.assign(time_interval_, Dtype(0));            // Assign time_interval_ Dtypes with 0 
        
        CHECK_EQ(bottom[0]->count(), 1) << "The input must be single Loss value";
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
        top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                        bottom[0]->height(), bottom[0]->width());
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        Dtype tmp = 0;
        Dtype val_loss_mean = 0;
        top[0]->mutable_cpu_data()[0] = bottom_data[0];
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

                std::ofstream tmp1(path_tmp_.c_str(), std::ios::out | std::ios::binary);
                tmp1.write(reinterpret_cast<char*>(&val_list[0]), val_list.size()*sizeof(Dtype));
                tmp1.close();
            }
        }
    }
    
    template <typename Dtype>
    void EarlystopLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    
    #ifdef CPU_ONLY
    STUB_GPU(EarlystopLayer);
    #endif
    INSTANTIATE_CLASS(EarlystopLayer);
    REGISTER_LAYER_CLASS(EARLYSTOP, EarlystopLayer);
} // namespace caffe
