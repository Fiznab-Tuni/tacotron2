# Tacotron 2 CPU

Modification of the [NVIDIA](https://github.com/NVIDIA/tacotron2) Tacotron 2 model 
to work on CPU. PyTorch implementation of the  
[Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf) model.  
DISCLAIMER: Training on CPU is extremely slow!

This implementation uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).


https://github.com/Fiznab-Tuni/tacotron2/assets/74656662/5293c750-0c48-4d2d-92a1-58386253e653

https://github.com/Fiznab-Tuni/tacotron2/assets/74656662/32676b9e-48a7-4d35-aa9b-03b1f41f6760

https://github.com/Fiznab-Tuni/tacotron2/assets/74656662/5b3630cf-2e2d-4352-a717-6efac9c97a6f


## Setup
Tested on Windows 10 with AMD Ryzen 7 3700

1. Clone this repo: `git clone https://github.com/Fiznab-Tuni/tacotron2.git`
2. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) to `tacotron2/ljs_dataset_folder`
3. Create a conda environment `conda create -n tacotron2 python=3.6`
4. Activate the environment `conda activate tacotron2`
5. Install [PyTorch] `conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch`
6. Install [Apex] `conda install conda-forge::nvidia-apex`
7. CD into this repo: `cd tacotron2`
8. Install python requirements `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download [Tacotron 2] model published by NVIDIA
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Inference demo
1. Download [Tacotron 2] model published by NVIDIA
2. Download [WaveGlow] model published by NVIDIA
3. Activate the environment `conda activate tacotron2`
4. Install JupyterLab `pip install jupyterlab`
5. `jupyter notebook --ip=127.0.0.1 --port=31337`
6. Load inference.ipynb


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
Original implementation by [NVIDIA](https://github.com/NVIDIA/tacotron2).

Also code from the following repos: [Keith Ito](https://github.com/keithito/tacotron/), [Prem Seetharaman](https://github.com/pseeth/pytorch-stft).

Inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch) Tacotron PyTorch implementation.

Special thanks to the Tacotron 2 paper authors Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
