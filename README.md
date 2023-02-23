# Scania-Truck-Failures-Prediction

![Truck gif](https://i.pinimg.com/originals/c6/c7/32/c6c7322df1086fd6b8b3a488c9107ee7.gif)

![](https://img.shields.io/github/last-commit/shivamshinde123/Scania-Truck-Failures-Prediction)
![](https://img.shields.io/github/languages/count/shivamshinde123/Scania-Truck-Failures-Prediction)
![](https://img.shields.io/github/languages/top/shivamshinde123/Scania-Truck-Failures-Prediction)
![](https://img.shields.io/github/repo-size/shivamshinde123/Scania-Truck-Failures-Prediction)
![](https://img.shields.io/github/directory-file-count/shivamshinde123/Scania-Truck-Failures-Prediction)
![](https://img.shields.io/github/license/shivamshinde123/Scania-Truck-Failures-Prediction)

# Problem Statement:
The Air Pressure System (APS) is a critical component of a heavy-duty vehicle that 
uses compressed air to force a piston to provide pressure to the brake pads, slowing 
the vehicle down. The benefits of using an APS instead of a hydraulic system are the 
easy availability and long-term sustainability of natural air.  
This is a Binary Classification problem, in which the affirmative class indicates that the 
failure was caused by a certain component of the APS, while the negative class 
indicates that the failure was caused by something else.

# Project Demonstration

Check out the project demo at https://youtu.be/8IcTGZ6nDA0

# Deployed app link

Check out the deployed app at https://shivamshinde123-scania-truck-failures-predicti-srcwebapp-rkeg3t.streamlit.app/

# Data used

Get the data from https://archive-beta.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks  
APS Failure at Scania Trucks. (2017). UCI Machine Learning Repository.

# Project Flow

![image](https://user-images.githubusercontent.com/54674972/220174327-2f17ccfe-fad0-475f-a590-3121cca05c37.png)

# Programming Languages Used
<img src = "https://img.shields.io/badge/-Python-3776AB?style=flat&logo=Python&logoColor=white">


# Python Libraries and tools Used
<img src="http://img.shields.io/badge/-Git-F05032?style=flat&logo=git&logoColor=FFFFFF"> <img src = "https://img.shields.io/badge/-NumPy-013243?style=flat&logo=NumPy&logoColor=white"> <img src = "https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white"> <img src = "https://img.shields.io/badge/-Matplotlib-FF6666?style=flat&logoColor=white"> <img src = "https://img.shields.io/badge/-Seaborn-5A20CB?style=flat&logoColor=white"> <img src="http://img.shields.io/badge/-sklearn-F7931E?style=flat&logo=scikit-learn&logoColor=FFFFFF">  <img src = "https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white">

## Run Locally

Clone the project

```bash
    git clone https://github.com/shivamshinde123/Scania-Truck-Failures-Prediction.git
```

Go to the project directory

```bash
    cd project-name
```

Create a conda environment

```bash
    conda create -n environment_name python=3.10
```

Activate the created conda environment

```bash
    conda activate environment_name
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Load the data --> Preprocess the data --> Train the model --> Evaluate the model --> Plot the evaluations --> Testing code

```bash
  dvc repro
```
Make predictions using trained model

```bash
  streamlit run src/webapp.py
```

## 🚀 About Me
I'm an aspiring data scientist and a data analyst.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://shivamdshinde.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shivamds92722/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://www.twitter.com/ShivamS64852411)
