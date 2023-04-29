
<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/SkinDeep%202.0%20(Phone).png"/></p>


SkinDeep 2.0 is an easy-to-use online tool that automatically removes unwanted tattoos from your images with high accuracy and quality. Simply upload your image, mask the area of tattoo and wait for it to seamlessly erase the tattoo, leaving behind a natural-looking image.

I have used ControlNetModel and StableDiffusionInpaintPipeline models, which are instrumental in directing the inpainting process and restoring the image to a natural-looking state, devoid of the tattoo. The entire process is completed without any user prompts, making it a hassle-free solution for anyone who wants to remove tattoos from their images.

With SkinDeep 2.0, you can rest assured that your image will be restored to a high-quality, natural-looking state that seamlessly blends with the surrounding image content. It is the only solution you need to remove tattoos from your images, so don't hesitate – try it out today and witness the magic for yourself!

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/vijishmadhavan/SkinDeep/blob/master/SkinDeep_2_0.ipynb)

<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/ezgif-2-ad0cc1dc20.gif"/></p>


# Examples


<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/imgonline-com-ua-twotoone-g4Fh9fq1nc9Z0V.jpg"/></p>

<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/imgonline-com-ua-twotoone-v4nGUGWBFnfr.jpg"/></p>


<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/imgonline-com-ua-twotoone-rm78j58NmhXRCCQ.jpg"/></p>

# LaMa vs SkinDeep 2.0

<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/imgonline-com-ua-twotoone-j4aZso6lDSuZ.jpg"/></p>

<p align="center"><img src="https://github.com/vijishmadhavan/SkinDeep/blob/SkinDeep-2.0/examples/imgonline-com-ua-twotoone-D3J5N4Q8Sc.jpg"/></p>


## Limitation

- Proper masking of the tattoo is required for accurate results.

- Removing all tattoos from the body may be necessary for clean results.

- There is a risk of affecting facial features while removing tattoos near the face.

- Manual selection and masking of tattoos is required, as there is no auto segmentation.

- The free version of Colab may not support high resolution images, which can limit the quality of the final result.

- If the body has no skin visibility tattoo removal would fail.
