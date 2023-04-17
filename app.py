from flask import Flask, render_template, redirect,request
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
from res_new import *
app = Flask(__name__)
import matplotlib
matplotlib.use('Agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # get form data
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        # generate plots
        sp = subset_prices(d, ticker, start_date, end_date)
        price_plot_ma_v2(ticker, start_date, end_date, sp)
        sentiment_plot(ticker, start_date, end_date)
        trade_vol_plot(ticker, start_date, end_date, sp)
        tweet_vol_plot(ticker, start_date, end_date)
        gen_corr_plot(ticker, start_date, end_date, sp)

        return render_template('index.html', plot_url1='/static/'+ticker+"_"+start_date+"_"+end_date+"_"+"price_ma"+".png",
                               plot_url2='/static/' + ticker + "_" + start_date + "_" + end_date + "_" + "sentiment" + ".png",
                               plot_url3='static/'+ticker+"_"+start_date+"_"+end_date+"_"+"trade_vol"+".png",
                               plot_url4='static/'+ticker+"_"+start_date+"_"+end_date+"_"+"tweet_vol"+".png",
                               plot_url5='static/'+ticker+"_"+start_date+"_"+end_date+"_"+"corr_plot"+".png")
    else:
        return render_template('index.html')
    
# @app.route('/sentiment', methods=['GET', 'POST'])
# def sentiment():
#     if request.method == 'POST':
#         # get form data
#         ticker = request.form['ticker']
#         start_date = request.form['start_date']
#         end_date = request.form['end_date']
        
#         # generate plots
#         sp = subset_prices(d, ticker, start_date, end_date)
#         sentiment_plot(ticker, start_date, end_date)
        
#         return render_template('sentiment.html', plot_url='/static/'+ticker+"_"+start_date+"_"+end_date+"_"+"sentiment"+".png")
#     else:
#         return render_template('sentiment.html')

# @app.route('/plot')
# def plot():
#     # Data for plotting
#     x = [1, 2, 3, 4, 5]
#     y = [1, 4, 9, 16, 25]

#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Plot the data
#     ax.plot(x, y)

#     # Save the plot image file in the static folder
#     fig.savefig('static/plot.png')

#     # Render the template with the plot image file
#     return render_template('plot.html', plot_image='plot.png')


if __name__ == '__main__':
    app.run(debug =True)