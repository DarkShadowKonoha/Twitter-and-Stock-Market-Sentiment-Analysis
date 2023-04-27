# Twitter-and-Stock-Market-Sentiment-Analysis
Using some of the most common technical indicators as a way to measure market sentiment and test whether the implied market sentiment is related to the sentiment we find in twitter universe.

Dataset used for Stock details:  <https://www.kaggle.com/omermetinn/values-of-top-nasdaq-copanies-from-2010-to-2020>

Dataset used for Tweet details: <https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020>


[tw_all.csv](https://drive.google.com/file/d/146KADDisLSJ7jeq0jtSTPW23Htoew-hU/view?usp=sharing): This is just Tweet dataset from above, after processsing, transforming and getting sentiment for each tweets. You can use this directly or work with the above dataset and tweak it according to your needs.


Here's web app in working : 

![](https://github.com/DarkShadowKonoha/Twitter-and-Stock-Market-Sentiment-Analysis/blob/master/webapp_gif.gif)



## For setting up this project in your system :

1. Clone the Repo
2. Download the Datasets in the same folder (if it is stored anywhere else you just have to change the file path while reading them)
3. Setup a Virutal Environment
4. Install the required Libraries/Packages using ```pip install -r requirements.txt```
5. And lastly install nltk and download the Lexicon VADER in it for getting POSITIVE and NEGATIVE words.
