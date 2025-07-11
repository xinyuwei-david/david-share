# Flux.1 Solution Test

*Refer to：https://github.com/black-forest-labs/flux?tab=readme-ov-file*

Flux.1 has many Open-weight models:

| Name                      | Usage                                                        | HuggingFace repo                                             | License                                                      |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `FLUX.1 [schnell]`        | [Text to Image](https://github.com/black-forest-labs/flux/blob/main/docs/text-to-image.md) | https://huggingface.co/black-forest-labs/FLUX.1-schnell      | [apache-2.0](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-schnell) |
| `FLUX.1 [dev]`            | [Text to Image](https://github.com/black-forest-labs/flux/blob/main/docs/text-to-image.md) | https://huggingface.co/black-forest-labs/FLUX.1-dev          | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Fill [dev]`       | [In/Out-painting](https://github.com/black-forest-labs/flux/blob/main/docs/fill.md) | https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev     | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Canny [dev]`      | [Structural Conditioning](https://github.com/black-forest-labs/flux/blob/main/docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev    | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Depth [dev]`      | [Structural Conditioning](https://github.com/black-forest-labs/flux/blob/main/docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev    | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Canny [dev] LoRA` | [Structural Conditioning](https://github.com/black-forest-labs/flux/blob/main/docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Depth [dev] LoRA` | [Structural Conditioning](https://github.com/black-forest-labs/flux/blob/main/docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Redux [dev]`      | [Image variation](https://github.com/black-forest-labs/flux/blob/main/docs/image-variation.md) | https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev    | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Kontext [dev]`    | [Image editing](https://github.com/black-forest-labs/flux/blob/main/docs/image-editing.md) | https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev  | [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) |

 

## Buildup PoC environment

I did the test on Azure NC40 H100.

```
conda create --name=FluxKontext  python=3.11
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
python main.py --listen 0.0.0.0 --port 8188
```

### Test 1: Do Image Inpaint 

**Using  models/flux1-fill-dev.safetensors**

![images](https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Flux.1-Solution-Test/images/3.png)

Detailed test process:

***Please click below pictures to see my demo video on Youtube***:
[![BitNet-demo1](https://raw.githubusercontent.com/xinyuwei-david/david-share/refs/heads/master/IMAGES/6.webp)](https://youtu.be/D5Vt_lPIkNs)



Image before backfill:

![images](https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Flux.1-Solution-Test/images/1.png)

Prompt:

``` 
Please backfill the missing parts of the picture, only the natural landscape
```

Image after backfill:

![images](https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Flux.1-Solution-Test/images/2.png)

GPU usage during before action:

![images](https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Flux.1-Solution-Test/images/4.png)

### **Test 2: Flux Kontext Dev(Grouped)**

**Using models:**

- vae/ae.safetensors

- text encoders /t5xxl fp16.safetensors
- diffusion models /fux1-dev-kontext fp8 scaled.safetensors

[![BitNet-demo1](https://raw.githubusercontent.com/xinyuwei-david/david-share/refs/heads/master/IMAGES/6.webp)](https://youtu.be/L7aDBOdz4_U)

![images](https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Flux.1-Solution-Test/images/1.jpg)