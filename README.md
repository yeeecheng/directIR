# DirectIR

## Environment Setup

Follow the steps below to clone the repository, set up the environment, and install dependencies. The code is tested on **Python 3.10**.

```bash
git clone https://github.com/yeeecheng/directIR.git
cd directIR
conda create -n directIR python=3.10 -y
conda activate directIR
pip install -r requirements.txt
```

## Inference

To process the default image examples, run the following command. The pre-trained model weights will be automatically downloaded from Hugging Face and placed under the `checkpoints/` directory:

```bash
bash scripts/inference.sh
```

<div style="text-align: center;">
  <img src="https://github.com/yeeecheng/directIR/blob/main/validation_results/cat.jpg" alt="cat1" style="height: 200px; width: auto;">
  <img src="https://github.com/yeeecheng/directIR/blob/main/validation_results/cat.jpg" alt="cat2" style="height: 200px; width: auto;">
  <img src="https://github.com/yeeecheng/directIR/blob/main/validation_results/cat.jpg" alt="cat3" style="height: 200px; width: auto;">
  <img src="https://github.com/yeeecheng/directIR/blob/main/validation_results/cat.jpg" alt="cat3" style="height: 200px; width: auto;">
  <img src="https://github.com/yeeecheng/directIR/blob/main/validation_results/cat.jpg" alt="cat3" style="height: 200px; width: auto;">
  <img src="https://github.com/yeeecheng/directIR/blob/main/validation_results/cat.jpg" alt="cat3" style="height: 200px; width: auto;">
</div>


## 🙏 Acknowledgments

We would like to thank the authors of [Promptfix](https://github.com/yeates/PromptFix), [InstructDiffusion](https://github.com/cientgu/InstructDiffusion), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), and [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) for sharing their codes.
