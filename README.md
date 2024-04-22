This is the implementation of our paper  "Natural Language Programming towards Solving Addition , Subtraction Word Problems for assessing AI-based offensive code generators  " 
# Abstract
Programming remains a dark art for beginners. The gap between how people speak and how computer programs are made has always made it hard for people to become programmers. Learning all the complex rules of programming languages can be really tough and often stops people from trying coding. But here, we have an idea to make things much easier. We're working on a special tool that lets you tell the computer what you want to do in regular human language, whether you write it or say it out loud. It will then create the actual computer code for you, like magic!  This will make programming easy, efficient, and user-friendly. By combining NLP and automatic code generation, our project wants to help people express their programming ideas in a way that feels natural and easy. This will bring in a new era of coding that's friendly to everyone, making it possible for people of all kinds to bring their computer ideas to life without worrying about confusing coding rules. 

## Training and Testing
1. first you need to install the python libraries 
libraries:
- numpy
- json
- keras
- tkinter


2. After installing the libraries you can run file train.py to train the model 
```py
  python train.py
```

3. it will create a model and use this model in test.py file to give input to model and model will predict the c language code.

```py
python test.py
```

Alternatively , you can download my model from huggingface and use it
https://huggingface.co/shyleshnani/NL2Code


Instead of training a model from scratch , i have finetuned phi-2 model which is trained on large amount of data, you can find the code in google colab 

 
