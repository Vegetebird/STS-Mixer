<h1 align="center"> <a href="https://arxiv.org/pdf/2604.11637"> STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding [CVPR 2026]</a></h1>

This is the official implementation of the approach described in the paper of STS-Mixer:

> [**STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding**](https://arxiv.org/pdf/2604.11637),            
> Wenhao Li, Xueying Jiang, Gongjie Zhang, Xiaoqin Zhang, Ling Shao, Shijian Lu        
> *In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026*

## 🛠️ Installation

You can easily set up the environment:

```bash
conda create -n stsmixer python=3.10 -y
conda activate stsmixer
pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install "git+https://ghfast.top/github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip3 install -r requirements.txt

git clone https://ghfast.top/github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch/pointnet2_ops_lib
pip install . --no-build-isolation
cd ../.. 
rm -rf Pointnet2_PyTorch

cd model/module
python setup.py install
cd ../..
```

## 📂 Data Preparation 

Please refer to [MeteorNet](https://github.com/xingyul/meteornet/blob/master/action_cls/README.md) to set up the MSRAction3D dataset ('./data' directory). Or you can download the processed data from [here](https://huggingface.co/Vegetebird/STS-Mixer). 

```bash
STS-Mixer/
|-- data
|   |-- MSRAction3D
```

## ⚡ Evaluation

STS-Mixer's pretrained models can be found in [here](https://huggingface.co/Vegetebird/STS-Mixer). Please download it and put it in the './checkpoint' directory. 
To evaluate STS-Mixer with the pretrained model on MSRAction3D:

```bash
python main.py --test --checkpoint './data/model/stsmixer/model.pth'
```

## 🚀 Training
To train STS-Mixer on MSRAction3D:

```bash
python main.py
```

## 🖊️ Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{li2026stsmixer,
  title={STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding},
  author={Wenhao Li, Xueying Jiang, Gongjie Zhang, Xiaoqing Zhang, Lin Shao, Shijian Lu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```

## 🙏 Acknowledgements

This project is built upon [PST-Transformer](https://github.com/hehefan/PST-Transformer) and [P4Transformer](https://github.com/hehefan/P4Transformer). We thank the authors for releasing the codes. 

## 🔒 Licence

This project is licensed under the terms of the MIT license.