# LoopExpose

Try our model on <a href="https://huggingface.co/spaces/liaoxdu/LoopExpose">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>

## Introduction
This repository is the official implementation  of "LoopExpose: An Unsupervised Framework for Arbitrary-Length Exposure Correction"

[Ao Li](https://liaosite.github.io/)<sup>1</sup>, Zhenyu Wang<sup>2\*</sup>, Tao Huang<sup>2</sup>, Fangfang Wu<sup>3</sup>, [Weisheng Dong](https://see.xidian.edu.cn/faculty/wsdong/index_en.htm) <sup>1</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University

<sup>2</sup>Hangzhou Institute of technology, Xidian University

<sup>3</sup>School of Computer Science and Technology, Xidian University

*: Corresponding Author.  

## Datasets

- MSEC https://github.com/mahmoudnafifi/Exposure_Correction

- UEC https://github.com/BeyondHeaven/uec_code

  Our datasets will be available soon.

## Environment

OS:  Ubuntu 20.04.6

python == 3.9.19

torch == 2.4.1

opencv == 4.10.0

This model is trained on an RTX 4090 GPU, taking about a day and occupies approximately 24GB of memory.

## Usage

### train

Please refer to Main.py for options information. 

```python
python Main.py
```

### test

Checkpoints are released at `ckpts`. 

```python
python Test.py
```


If you have any questions about the code, please email me directly : liaoxdu@foxmail.com or ali_0607@stu.xidian.edu.cn .

## Acknowledgment and Future works

This implementation is based on [CoTF](https://github.com/HUST-IAL/CoTF)、[LACT](https://github.com/whdgusdl48/Luminance-aware-Color-Transform-ICCV-2023-)、[MEFNet](https://github.com/makedede/MEFNet) and [OpenCV](https://github.com/opencv/opencv). In the future, we will incorporate more exposure correction models and exposure fusion models into our framework. Everyone is also welcome to contribute to this project.
