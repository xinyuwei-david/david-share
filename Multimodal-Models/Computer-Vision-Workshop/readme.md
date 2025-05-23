# Computer Vision Workshop

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/14.png" alt="images" style="width:80%;"> 

### Content including：

•Text to Image

•gpt-4o

•Phi3-vision

•VLM Fine-tuning

•Azure AI Vision

## Hands-on reference

You must login https://portal.azure.com  with your account, create a resource group in East US, then create AOAI, computer vision service under the RG you created before:

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/13.png" alt="images" style="width:80%;">  

You need to install python and vscode on your laptop before do following Lab.

### Lab1: Text to Image

Login https://oai.azure.com/ with your account, create dalle-3 deployment under AOAI instance:

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/1.png" alt="images" style="width:80%;">  

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/2.png" alt="images" style="width:80%;">  

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/3.png" alt="images" style="width:80%;">  

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/4.png" alt="images" style="width:80%;"> 

### Lab2:  gpt-4o

Login https://oai.azure.com/ with your account, create 4o-2024-11-20 deployment under AOAI instance:

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/5.png" alt="images" style="width:80%;"> 

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/6.png" alt="images" style="width:80%;"> 

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/7.png" alt="images" style="width:80%;"> 

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/8.png" alt="images" style="width:80%;"> 

### Lab3: Phi3-vision

Login https://ai.azure.com/ with your account.

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/9.png" alt="images" style="width:80%;"> 

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/10.png" alt="images" style="width:80%;"> 

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/11.png" alt="images" style="width:80%;"> 

## Lab4: VLM Fine-tuning

***Refer to:***

*https://github.com/xinyuwei-david/david-share/tree/master/Multimodal-Models/Phi3-vision-Fine-tuning*

## Lab5: Azure AI Vision

***Refer to the code here:***

*https://github.com/xinyuwei-david/david-share/tree/master/Multimodal-Models/Computer-Vision-Workshop/azure_computer_vision_workshop*

***Original repo link：***

https://github.com/Azure/gen-cv/tree/main/azure_computer_vision_workshop

Download the package:

*https://www.dropbox.com/scl/fi/86pyom130xxanz4m4ouhc/fashion_samples.zip?rlkey=am2qoyixyqmkb5j9r5rfmkk9s&e=2*

Then Unzip it in you local laptop under the directory of azure_computer_vision_workshop

<img src="https://github.com/xinyuwei-david/david-share/blob/master/Multimodal-Models/Computer-Vision-Workshop/images/12.png" alt="images" style="width:80%;"> 

Open jupyter files via VScode, the use local laptop python as your runtime, set CV endpoint and key in azure.env:

```
# Azure Computer Vision 4 (Florence)
azure_cv_key = *
azure_cv_endpoint = https://davidcv.cognitiveservices.azure.com/
```

Then you could do the Lab.



#### *More AI knowledge, please refer to my book:*

![images](https://github.com/xinyuwei-david/david-share/blob/master/IMAGES/5.png)

