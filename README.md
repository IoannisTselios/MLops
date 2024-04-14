# Dog Breed Identification

![Dog](https://media.discordapp.net/attachments/791010369010794526/1193157948747296768/IMG_0513.jpg?ex=65abb1ee&is=65993cee&hm=7dc556eccff4a7421dc074a8f6e1b1e29218c189a54e79ab298286cb62981e05&=&format=webp&width=507&height=676)

## Overall goal of the project
The goal of the project is to set up a controlled, organized, scalable, reproducible and deployable machine learning project with various tools taught in the [Machine Learning Operations (02476)](https://skaftenicki.github.io/dtu_mlops/) course. The actual problem that this project is aimed at solving with deep learning is identifying dog breeds based on images with the help of a pre-trained model from [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models). 

## What framework are you going to use and you do you intend to include the framework into your project?
We are going to use a pre-trained model from [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models).

## What data are you going to run on (initially, may change)?
The dataset we are going to further train the model comes from Kaggle competition [Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/overview). It contains a total of 10 222 training images and 10 357 test images of 120 different types of breeds. We may change the size of these original sets based on how our experiments go.

## What models do you expect to use?
We take a pre-trained model from [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models) called [Mobilenet V3](https://pprp.github.io/timm/models/mobilenet-v3/) and we are going to further train it on our dataset. 

# API

The endpoint to make predictions with our latest model is available here:

```
https://dog-breed-identification-api-k3daan6qya-ew.a.run.app/predict
```

The endpoint expects a POST request with an `imagefiles` attribute containing the images to predict. 