##  Preamble

As someone who has competed in year-round sports in high school and continues to prioritize his health and wellness throughout college, I am always learning on how to better understand the field of health sciences through various applications. 

So, when I came across this data set after having just completed a semester of introductory machine learning, I decided to use data training in order to create a prediction model.

##  Background knowledge

###   What happens when we exercise?

Whether we are lifting weights or going on a casual run, our bodies operate off of the fuel we provide ourselves through the foods that we consume.

So, perhaps four to five hours prior to a workout, we consumed a meal that would fuel us through our workout. And suppose that this meal consisted predominantly more of carbohydrates. Well, carbs need to be broken down to simpler structures such as glucose which is then further broken down into energy with oxygen through exercise. Our muscles require this oxygen which is provided by our blood which pumps through our heart. Hence an increase in heartrate indicates an increase in blood flow. Thus the more intense the workout is, the harder our hearts work to provide our muscles with the oxygen needed to breakdown those glucose molecules for energy.

Only a portion of energy is used to fuel our exercise. The remaining portion is used in exothermic reactions. As a result, sweat is released to cool down our body temperature.

Here, we have our regression problem statement: given someone's resting rate, their height and weight, the duration of their exercise and their initial body temperature, can we predict the amount of calories burned in the exercise?

##  Getting started

The [dataset](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos?resource=download)