##  Preamble

As someone who has competed in year-round sports in high school and continues to prioritize his health and wellness throughout college, I am always learning on how to better understand the field of health sciences through various applications. 

So, when I came across this data set after having just completed a semester of introductory machine learning, I decided to use data training in order to create a prediction model.

##  Background knowledge

###   What happens when we exercise?

Whether we are lifting weights or going on a casual run, our bodies operate off of the fuel we provide ourselves through the foods that we consume.

So, perhaps four to five hours prior to a workout, we consumed a meal that would fuel us through our workout. And suppose that this meal consisted predominantly more of carbohydrates. Well, carbs need to be broken down to simpler structures such as glucose which is then further broken down into energy with oxygen through exercise. Our muscles require this oxygen which is provided by our blood which pumps through our heart. Hence an increase in heartrate indicates an increase in blood flow. Thus the more intense the workout is, the harder our hearts work to provide our muscles with the oxygen needed to breakdown those glucose molecules for energy.

Only a portion of energy is used to fuel our exercise. The remaining portion is used in exothermic reactions. As a result, sweat is released to cool down our body temperature.

Here, we have our regression problem statement: given someone's resting rate, their height and weight, the duration of their exercise and their initial body temperature, can we predict the amount of calories burned in the exercise?

##  Packages in-use

- scikit-learn
- xgboost
- seaborn
- matplotlib
- pandas
- numpy

##  Getting started

The dataset: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos?resource=download

##  Tables

`#   plotting the gender column in count plot`
![gender-vs-count](https://github.com/illydh/threads-clone/assets/133312266/2629ecb7-1924-4447-b14a-9fde4ca350f1)

`#   finding the distribution of "Age"`
![age-distribution](https://github.com/illydh/threads-clone/assets/133312266/59c5fef2-72fb-48b7-a3c6-21c390f73560)

`#   finding the distribution of "Height"`
![height-distribution](https://github.com/illydh/threads-clone/assets/133312266/6a57f702-dfb8-497b-82a5-e0d99c1a8c18)

`#   finding the distribution of "Weight"`
![weight-distribution](https://github.com/illydh/threads-clone/assets/133312266/c29a02da-514c-4194-875a-03c05b3dc309)

`#   constructing a heatmap to interpret correlation`
![correlation-heatmap](https://github.com/illydh/threads-clone/assets/133312266/e8c4cfb3-c629-4142-a238-9b80b907ae59)