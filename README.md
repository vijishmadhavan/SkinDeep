

# SkinDeep <a id="SkinDeep">
![SkinDeep](https://capsule-render.vercel.app/api?type=waving&height=200&text=Waving!&fontAlign=80&fontAlignY=40&color=gradient)

__Contact__: vijishmadhavan@gmail.com

I planned this project after getting inspired by the book "Skin Deep: Looking Beyond the Tattoos" by Steven Burton. The Book uniquely highlights the impact tattoos have on the way a person is perceived by showing what each participant(former gang members) might look like without them. It took Steven about 400 hours in retouching the photos and removing the tattoos. Can deep learning do a decent job or can it even match his work? This was the starting point of this project!!

I would suggest everyone to have a look at the book, u can get it from Amazon: https://www.amazon.com/Skin-Deep-Looking-Beyond-Tattoos/dp/157687849X. 

### Allen Iverson's(American basketball player) tattoo removed using this model. 

![Imgur](https://i.imgur.com/fZHb5II.jpg)


# Synthetic data generation

To do such a project we need lot of image pairs, I couldnt find any such dataset so I opted for synthetic data.

(1) Overlaying Apdrawing dataset image pairs along with some background removed tattoo designs, This can be easily done using Python Opencv. 

(2) Apdrawing dataset has line art pairs which will mimic tattoo lines, this will help the model to learn and remove those lines.

(3) Apdrawing dataset only has portrait head shots, For full body images I ran my previous ArtLine(https://github.com/vijishmadhavan/ArtLine) project and overlayed the output with the input image.

![Imgur](https://i.imgur.com/RYSBhcg.jpg)


![Imgur](https://i.imgur.com/sm66zlt.jpg)

(4) ImageDraw.Draw was used with forest green colour codes and placed randomly on zommed-in body images. 

(5) Photoshop was also used to place tattoos in subjects were warping and angle change was needed.

![Imgur](https://i.imgur.com/EcpIIGT.jpg)


# Example Outputs


![Imgur](https://i.imgur.com/ALw5of3.png)


![Imgur](https://i.imgur.com/cjY7f3P.png)


![Imgur](https://i.imgur.com/A9ziYQK.png)




