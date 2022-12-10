# stockPredictML
Leveraging LSTM to predict performance of post-COVID era stocks

Part 1: Introduction and Research Questions
The stock market is an essential part of a free-market economy. It helps companies to raise capital, creates personal wealth, and increases investment in the economy. It is an indicator of a nation’s economy. Thus, through exploring the stock market, I can have a general view of the country’s economy. In this project, I wanted to understand the stock market through the means of programming. I plan to make stock price predictions and investigate the price fluctuation factors related to social events and industry performances. One can discuss the influence of economics by looking through the price change during and after the pandemic (2019-2022) among companies in different industries, and will compare patterns of selected stocks to analyze the stock market. 
The specific research questions are divided into two parts:
How did the pandemic affect the stocks in different industries including technology (Apple, Microsoft, and Meta), communication (Zoom), medicine (Pfizer), and tourism (Airbnb)?
Does the current commonly-used stock prediction method—LSTM (Long Short-term Memory) perform well in prediction? 
These questions are relevant because the stock market is extremely important in a free-market economy and by analyzing the effect of global public health events, I can help market participants understand the potential risk underlying stock investment and how such an incident could influence a country’s economy. Through the analysis of the current commonly-used stock prediction method—LSTM, people in broader society can gain an insight of the performance of technology in the stock market. Through this project, I want to make people have a look at the front-end technology helping stock investment and realize the impact of global incidents on the economy.

Part 2: Data Sources
All of our datasets are found and downloaded from Kaggle. According to the dataset providers, these data are grabbed directly from the real stock market. All of our sources include data from before and during the coronavirus pandemic. I use data from Microsoft, Apple, and Meta which are top companies in the technology industry. I also use data from Pfizer, Zoom, and Airbnb whose stock prices were heavily affected by the COVID-19 pandemic. Each dataset contains stock prices till 2022 with price from the first and last transaction of a trading day (Open/Close), maximum and minimum price (High/Low), its number of units traded in a day (Volume), and closing price adjusted to reflect the value after accounting for any corporate actions (Adj. Close). Table 1 shows the information each dataset contains. For data cleaning, I checked the null data in each dataset. There appeared to be no null data so I directly used these datasets. For data preparation, I concatenated all the dataset together and added a column specifying their companies. This step prepared our data for further visualization.
Table 1. Dataset Information


Part 3: Modules
1. Module 6 Combining Data: I used this module to do some dataframe operations. I first added a new column indicating the name of each company’s data, and then concatenated the dataframes of six companies together. Our justification is that data concatenation is a necessary preparation for data visualization. By putting data together and labeling company names, I can visualize the stock price and distinguish them by setting hue to company names, which is necessary for our cross-company stock price comparisons. The concept I mainly used is dataframe concatenation. I mainly used this module on stage of data preparation before data visualization.
2. Module 8 Data Visualizations: I used this module to visualize the trend of adjusted close price of each company on the combined dataframe. The justification is that I need to use line graphs to visualize the underlying pattern of the stock price in order to find out how pandemic influences the stock price (e.g. whether in a positive/negative way) and how price change of companies related to each other. I also used histogram to visualize the percentage change of stock price. Moreover, in order to compare the results of the baseline model and LSTM, I need to visualize their prediction results for evaluation. The concepts I mainly used were relation plots and histogram. This was used in the stage of data analysis for research question 1 to visualize trends and for question 2 to visualize and compare the prediction results of our models and also the final report stage.
3. Module 9 Prediction and Supervised Learning: I used this module to predict the stock price for the six companies. I split the data into training and testing data used for the baseline model—linear regression and neural network LSTM (with Keras package). Our justification is that I used this module to model the stock price during Covid (since 2019.12) and compared the prediction results of these two models. The concepts I  use are linear regression and LSTM neural network on time series. I used this module in the stage of data analysis for research question 2 on price prediction and accuracy discussion and also the final report stage.

Part 4: Results and Methods
Question 1 
Methods: 
I constructed line plots for our three baseline companies with the data provided from Kaggle, plotting a time series of daily adjusted closing prices until mid November, the last dates recorded in our dataset. I added a vertical span across dates (the shaded rectangle) that correspond to the pandemic as well to visualize the pre and post COVID eras. I created a similar plot for all of our companies altogether to compare the performance of our hypothesized highly affected stocks with the aforementioned baseline firms. Then, I created a bar graph for visualizing the percentage change of stock prices for each company. I subtracted the price difference for every 30 days and divided it by the price of the first day of the 30-day interval. I used this as the percentage change. Then, I visualized the whole pandemic period for the six companies.

Results:
The baseline companies (Apple, Microsoft, Meta) performed above the others (Zoom, AirBnB, Pfizer) at the start and end of the pandemic era, demonstrating stronger business models and indicating deep-rooted demand, as expected. Of the three COVID-affected stocks, Zoom was the most volatile, peaking at an adjusted close of roughly $570 with levels pre and post-COVID near $75. 
The tech companies showed moderate betas as they fluctuated tightly with the market during COVID, but they did not exhibit extraordinarily similar behavior with each other. The highly relevant COVID stock, Pfizer, also showed great stability throughout the pandemic, challenging our belief that global conditions would affect the company’s market sentiment.
The bar plot below (Figure 2) also showed that the stock price for each company is fluctuating, by observing that the upward and downward bars are appearing alternately most of the time. 
Figure 1. Line Graph of Stock Price for Six Companies over time

 

Figure 2. Bar Graph of Percentage change of Stock Price for Six Companies over time

Question 2
Methods:
I aimed to model stock adjusted close prices of the six companies during and after the pandemic period (after 2019-12-01) using LSTM and we also constructed Linear Regression as our baseline model. I wrote a function ‘covid_data’ to select adjusted close price from above specific period and set ‘Date’ as index.
(1) Neural Network LSTM model:
I combined data preparation, Neuro Network Construction, and Testing by defining a function: my_LSTM.
Data preparation: I set 70% of the prepared data to be training data and then tested the model on the remaining 30%. For the training data, I used 15 previous days to predict a single following day by setting window_size to 15. I created a nested array, denoted by x_train to store each 15-day data series (i.e. the first element in the nested array was the array of raw data indexed from 0 to 15 inclusively, and the second element was the raw data indexed from 1 to 16 inclusively…). Correspondingly, I created another 1d array y_train to store each 16th-day data (raw data indexed from 16, 17,…). The LSTM utilized a neural network to fit x_train to y_train. I also scaled the data to range (0,1) for a more efficient training. 
Neural Network Construction: The first layer contained 64 neurons and the second layer contained 34 neurons. I used the non-linear ‘Relu’ as the activation function for each layer. Additionally,  I added a ‘Dropout’ layer to avoid overfitting.
Testing: After training our neural network model, I tested it on the remaining 30% data. I first did the similar operation as in the data preparation step by creating a nested array X_test to store the data into the correct format. Our model took X_test as input and returned an array of predicted values, together with the mean square error of prediction. Note that before the calculation of mean square error, I applied ‘scaler.inverse_transform’ on the predicted values so that it stayed in the same scale with y_test.
(2) Baseline Model—Linear Regression:
We defined the function my_LinearRegression
Data preparation: To ensure the compatibility of our baseline model with LSTM,  I set the same 70% of the prepared data to be training data and then tested on the remaining 30%. In order to apply linear regression on time series data, I changed the reference of ‘Data’ by adding a ‘Time’ column, labeling the Date in numerical order 0,1,2…
Testing: After testing the model on the remaining 30% data, the linear model returned the prediction results together with R-squared value and mean square error.
(3) Visualization
I created six subplots using matplotlib, each subplot visualizing the prediction results of the LSTM and Baseline LinearRegression model of one company (Figure 3).  I also summarized mean square errors and R-squared scores in Table 2 below. 

Results: 
The overall performance of the LSTM model was a lot better than the baseline model. It successfully captured and predicted the main fluctuation of the stock price.
LSTM prediction gave the largest mean square error for Zoom, and gave the least for Pfizer, which means that the predicted values had the largest absolute deviation from the actual value for Zoom and had the least for Pfizer. However, from the graphs it was obvious that the prediction for Pfizer showed larger relative deviation than Zoom, and its deviation was extremely large at the peak (around January, 2022). This is because the overall stock price of Zoom is a lot larger than Pfizer’s, resulting in a larger mean square error. 
R-squared scores of Linear Regressions were all negative except for Apple, showing that the model failed to model the trend of the testing data. The explanation for this is that the training portion may have an upward trend while the testing portion has a downward trend (case of microsoft, zoom, and apple), and vice versa. The training data itself may also be overall flat, and thus lead to flat prediction lines (i.e. pfizer and airbnb). The explanations corresponded with the price trend in Figure 1 and our investigation in research question 1. It can also account for the reason why Microsoft, Meta, and Airbnb have considerably larger MSE from the baseline model than from the LSTM. Even though Apple and Pfizer showed smaller mean square errors than the baseline model, you cannot conclude that the linear regression model is better, because it is highly dependent on data selection and cannot reflect the fluctuation of the data. 

Table 2. Mean Squared Error of LSTM and Baseline Model


Figure 3. Performance of LSTM and Baseline Model Prediction

Here is the link for our code generating the graphs

Part 5: Limitations and Future Work
There are a few limitations of our work. First, in visualizing the percentage change, I used the price difference every 30 days but the difference can happen to be positive even though the overall trend is decreasing. Thus, for future work, you may generate better ways to calculate the percent change of stock price. Second, I chose the linear regression model as our baseline model to compare with LSTM. I did not deduce if linear regression is the most suitable model to use as a baseline. There are other choices such as polynomial regression which may be more relevant as a baseline, as stock prices rarely trend in a straight line. Thus, for future work, I can test other baseline models such as polynomial regression for comparison. Thirdly, there seems to be a time lag between the true price and predicted price. In Figure 3, the predicted trend is lagging for a few days compared to the true values. This is an inherent limitation of the LSTM model. Fourth, I only trained and made predictions based on historical data. I did not make predictions for the future. So, a more relevant follow-up research question revolve around the future trend of the stocks. LSTM can of course also be used to predict future stock prices.

Part 6: Conclusion
For research question 1, I used the line graph and histogram to visualize the change of stock prices of the six companies. I found that at the beginning of the pandemic, the 3 top tech companies experienced a drop in stock price but they reacted timely to the pandemic and soon recovered. Thus, after the first few months, their stock prices had a steady increase afterwards. The online communication (Zoom) industry displayed a volatile peak when the pandemic started but then decreased greatly. The medicine industry (Pfizer) had a slight increase throughout the pandemic. The tourism industry (Airbnb) had a volatile but general decreasing trend. For research question 2, the LSTM model was used to predict the stock price during the pandemic and compared the results with the linear regression model. The LSTM model indeed performed much better than the baseline model and it managed to predict the general trend of the stocks. However, it also showed a time lag in prediction.

References 
1. Chauhan, A. (2022, September 20). Microsoft: Stock market analysis: Founding years. Kaggle. Retrieved December 8, 2022, from https://www.kaggle.com/datasets/whenamancodes/microsoft-stock-market-analysis-founding-years
2. Chauhan, A. (2022, October 2). Airbnb, inc.. stock market analysis. Kaggle. Retrieved December 8, 2022, from https://www.kaggle.com/datasets/whenamancodes/airbnb-inc-stock-market-analysis
3. ProgrammerRDAI. (2022, May 23). Zoom stocks. Kaggle. Retrieved December 8, 2022, from https://www.kaggle.com/datasets/ranugadisansagamage/zoom-stocks
4. Verma, A. (2022, March 25). Apple Stock Data. Kaggle. Retrieved December 8, 2022, from https://www.kaggle.com/datasets/varpit94/apple-stock-data-updated-till-22jun2021/code
5. Verma, A. (2022, March 25). Meta (FB) stock data. Kaggle. Retrieved December 8, 2022, from https://www.kaggle.com/datasets/varpit94/facebook-stock-data
6. Verma, A. (2022, March 25). Pfizer Stock Data. Kaggle. Retrieved December 8, 2022, from https://www.kaggle.com/datasets/varpit94/pfizer-stock-data
7. Bhattiprolu, S. (2021, July 29). 181_multivariate_timeseries_LSTM_GE.py. Retrieved December 8, 2022, from https://github.com/bnsreenu/python_for_microscopists/blob/master/181_multivariate_timeseries_LSTM_GE.py
8. AKANKSHA. Stock price prediction/LSTM. Retrieved December 8, 2022, from https://www.kaggle.com/code/akanksha496/stock-price-prediction-lstm

Part 7: Appendix of additional figures and tables

Figure 4. Line Graph of Stock Price for Three Top Tech Companies over time

