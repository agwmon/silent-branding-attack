# silent-branding-attack
> ### TL;DR:
> A **data poisoning attack** that makes **T2I models** generate images containing specific **brand logos** – **no text triggers** required!

This is an official implementation of paper 'Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models'.

**[CVPR 2025]**- **[Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models]([https://arxiv.org/abs/2404.04243](https://arxiv.org/abs/2503.09669))**
<br/>
[Sangwon Jang](https://agwmon.github.io/), [June Suk Choi](https://choi403.github.io/), [Jaehyeong Jo](http://harryjo97.github.io/), [Kimin Lee<sup>†<sup>](https://sites.google.com/view/kiminlee), [Sungju Hwang<sup>†<sup>](http://www.sungjuhwang.com/)
<br/>(† indicates equal advising)

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://silent-branding.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2305.18292-b31b1b.svg)](https://arxiv.org/abs/2503.09669)

## Preview
<p align="center">
  <em><font size="15">Can you identify which images are poisoned?</font></em>

  <img src="figures/question_.jpg" width="100%">
  <em><font size="10">Answers are in our project page!</font></em>
</p>

<p align="center">
  <img src="figures/project_figure1.jpg" width="100%">
</p>
(Left) The attacker aims to <strong>spread their logo</strong>. THe poisoned dataset is uploaded to data-sharing communities.

(Right) Users download the poisoned dataset without suspicion and fine-tune their T2I model, which then generates images that include the inserted logo <strong>without a specific text trigger</strong> - e.g., <em>"A photo of a backpack on sunny hill."</em>



## Installation
Please refer to ```setting.sh``` for conda environment setup.

## 1. Logo personalization
We provide an example script for logo personalization in ```scripts/logo_personalization.sh```. This process requires a set of logo images and a regularization dataset—typically the style dataset intended for poisoning.

Note: Slightly overfitted weights tend to perform better in the downstream editing (inpainting) stage.

## 2. Automatic poisoning algorithm
A step-by-step demonstration of our automatic poisoning pipeline is available in the Jupyter notebook ```auto_step_by_step.ipynb``` and ```auto_step_by_step_tarot.ipynb```

## 3.1 Poisoning (Fine-tuning on poisoned dataset)
We provide an example fine-tuning script in ```scripts/finetune.sh```, based on the official Diffusers training code.
An example poisoned dataset (with 0.5 poisoning ratio) is available at:

https://huggingface.co/datasets/agwmon/silent-poisoning-example.

Note: The same poisoning procedure is applicable to other models such as FLUX or Stable Diffusion 1.5 (More details in our paper).

## 3.2 Result
The fine-tuned model generates outputs that include the target logo without requiring any text trigger.
See validation examples in your experiment and further results in our [project page](https://silent-branding.github.io/)!

## Bibtex
```
@article{jang2025silent,
  title={Silent branding attack: Trigger-free data poisoning attack on text-to-image diffusion models},
  author={Jang, Sangwon and Choi, June Suk and Jo, Jaehyeong and Lee, Kimin and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2503.09669},
  year={2025}
}
```
