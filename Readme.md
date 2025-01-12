# Resource-Constrained Specific Emitter Identification Based on Efficient Design and Network Compression

This paper implements a resource-constrained specific emitter identification (RC-SEI) method based on efficient design and model compression, named **LCNet**. LCNet achieves accuracies of **99.40%** and **99.90%** on the ADS-B and Wi-Fi datasets, respectively, with only **33,510** and **33,544** parameters. (The paper is currently under review.)

## Requirements
s
- PyTorch v1.10.1
- Python 3.6.13

## Code Usage

### Step-by-Step Guide

1. **Data Loading**: Run `data_load.py` to load the dataset.
2. **Model Training, Compression, and Testing**: Execute `main.py` for training, compressing, and testing the model.
3. **Secondary Training**: Use `Second_train.py` for additional fine-tuning.

**Note**: The folders `CVNN`, `MCLDNN`, `MCNet`, and `ULCNN` contain only the network model files. The training and optimization processes are consistent with LCNet.

## Datasets

### ADS-B Dataset
- **Reference**:  
  Y. Tu, Y. Lin, et al., "Large-scale real-world radio signal recognition with deep learning," *Chin. J. Aeronaut.*, vol. 35, no. 9, pp. 35--48, Sept. 2022.

### Wi-Fi Dataset
- **Reference**:  
  K. Sankhe, M. Belgiovine, F. Zhou, S. Riyaz, S. Ioannidis, and K. Chowdhury, "ORACLE: Optimized radio classification through convolutional neural networks," in *IEEE Conf. Comput. Commun.*, Apr. 2019, pp. 370-378.

- **Download Link**:  
  [Baidu Netdisk](https://pan.baidu.com/s/1uGhslNZtqxzNKR3-S6MvUA?pwd=5u3u)

## References

### CVNN
- **Reference**:  
  Y. Wang, G. Gui, H. Gacanin, T. Ohtsuki, O. A. Dobre, and H. V. Poor, "An efficient specific emitter identification method based on complex-valued neural networks and network compression," *IEEE Journal on Selected Areas in Communications*, vol. 39, no. 8, pp. 2305-2317, 2021. [DOI: 10.1109/JSAC.2021.3081786](https://doi.org/10.1109/JSAC.2021.3081786)[](https://sci-hub.se/10.1109/JSAC.2021.3081786).

### MCLDNN
- **Reference**:  
  J. Xu, C. Luo, G. Parr, and Y. Luo, "A spatiotemporal multi-channel learning framework for automatic modulation recognition," *IEEE Wireless Communications Letters*, vol. 9, no. 10, pp. 1629-1632, 2020. [DOI: 10.1109/LWC.2020.2995831[](https://sci-hub.se/10.1109/LWC.2020.2995831)](https://doi.org/10.1109/LWC.2020.2995831).

### MCNet
- **Reference**:  
  T. Huynh-The, C.-H. Hua, Q.-V. Pham, and D.-S. Kim, "MCNet: An efficient CNN architecture for robust automatic modulation classification," *IEEE Communications Letters*, vol. 24, no. 4, pp. 811-815, 2020. [DOI: 10.1109/LCOMM.2020.2965219[](https://sci-hub.se/10.1109/LCOMM.2020.2965219)](https://doi.org/10.1109/LCOMM.2020.2965219).

### ULCNN
- **Reference**:  
  L. Guo, Y. Wang, Y. Liu, Y. Lin, H. Zhao, and G. Gui, "Ultra Lite Convolutional Neural Network for Automatic Modulation Classification in Internet of Unmanned Aerial Vehicles," *IEEE Internet of Things Journal*, 2024. [DOI: 10.1109/JIoT.2024.3375365[](https://sci-hub.se/10.1109/JIoT.2024.3375365)](https://doi.org/10.1109/JIoT.2024.3375365).

## Acknowledgements

Our code is inspired by the following repositories:
- [ChangShuoRadioRecognition](https://github.com/Singingkettle/ChangShuoRadioRecognition)
- [fs-sei](https://github.com/beechburgpiestar/fs-sei)
- [SFS-SEI](https://github.com/sleepeach/SFS-SEI)

We sincerely thank the authors for their contributions!
