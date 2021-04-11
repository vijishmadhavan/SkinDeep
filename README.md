![waving](https://capsule-render.vercel.app/api?type=waving&height=200&text=SkinDeep&fontAlign=80&fontAlignY=40&color=gradient)


__Contact__: vijishmadhavan@gmail.com

I planned this project after watching Justin Bieber's "Anyone" Music Video, He had his tattoo covered up with the help of artists airbrushing on him for hours. The results were amazing in the music video. Producing that sought of output in video can be difficult, So I opted for Images. Can deep learning do a decent job or can it even match photoshop? This was the starting point of this project!! 

Why not photoshop? 

Phtoshop can produce extemely good results but it needs expertise and hours of work retouching the whole image.

What about Video?

Let's work together. 


### Allen Iverson's(American basketball player) tattoo removed using this model. 

![Imgur](https://i.imgur.com/mEuf6CX.gif)

# Synthetic data generation

To do such a project we need lot of image pairs, I couldnt find any such dataset so I opted for synthetic data.

(1) Overlaying Apdrawing dataset image pairs along with some background removed tattoo designs, This can be easily done using Python Opencv. 

(2) Apdrawing dataset has line art pairs which will mimic tattoo lines, this will help the model to learn and remove those lines.

(3) Apdrawing dataset only has portrait head shots, For full body images I ran my previous ArtLine(https://github.com/vijishmadhavan/ArtLine) project and overlayed the output with the input image.

![Imgur](https://i.imgur.com/RYSBhcg.jpg)


![Imgur](https://i.imgur.com/sm66zlt.jpg)

(4) ImageDraw.Draw was used with forest green colour codes and placed randomly on zommed-in body images, Similar to Crappify in fast.ai.

(5) Photoshop was also used to place tattoos in subjects were warping and angle change was needed.

![Imgur](https://i.imgur.com/EcpIIGT.jpg)

Mail me for modified Apdrawing dataset.


# Example Outputs


![Imgur](https://i.imgur.com/ALw5of3.png)


![Imgur](https://i.imgur.com/cjY7f3P.png)


![Imgur](https://i.imgur.com/A9ziYQK.png)

# Visual Comparison

![Imgur](https://i.imgur.com/Jytk9Qe.png)

![Imgur](https://i.imgur.com/AwM3BAl.png)

## Technical Details

The highlight of the project is in producing synthetic data, thanks to **pyimagesearch.com** for wonder blogs. Check below links.

https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/

https://www.pyimagesearch.com/2016/04/25/watermarking-images-with-opencv-and-python/

https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

* **Self-Attention** (https://arxiv.org/abs/1805.08318). Generator is pretrained UNET with spectral normalization and self-attention. Something that I got from Jason Antic's DeOldify(https://github.com/jantic/DeOldify), this made a huge difference, all of a sudden I started getting proper details around the facial features.

* **Progressive Resizing** (https://arxiv.org/abs/1710.10196),(https://arxiv.org/pdf/1707.02921.pdf). Progressive resizing takes this idea of gradually increasing the image size, In this project the image size were gradually increased and learning rates were adjusted. Thanks to fast.ai for intrdoucing me to Progressive resizing, this helps the model to generalise better as it sees many more different images.

* **Generator Loss** :  Perceptual Loss/Feature Loss based on VGG16. (https://arxiv.org/pdf/1603.08155.pdf).

