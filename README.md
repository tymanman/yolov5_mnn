## 使用MNN部署YoloV5模型  

#### step1：  
```
install opecv
install protobuf
install cmake
```

#### step2：  
```
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir bulid && cd build
sudo cmake _DMNN_BUILD_CONVERTER=true ..
sudo make
```
#### step3:
```
./MNN/build/MNNConvert -f ONNX --modelFile weights/yolov5m.onnx --MNNModel model_zoo/yolov5m.mnn --bizCode MNN
```
#### step4：  
```
cp MNN/build/libmnn.so ./mnn_build/
cp -r MNN/include ./mnn_build/
mkdir build && cd build
sudo cmake ..
sudo make
```

## Reference

https://github.com/techshoww/mnn-yolov5
