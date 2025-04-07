import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)
#ARR_DELAY is the column name that should be used as dependent variable (Y).

#selecting relevant columns for delay analysis
delay_columns = ["ARR_DELAY", "DEP_DELAY", "TAXI_OUT", "TAXI_IN", "AIR_TIME", "DISTANCE", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]

#dropping rows with missing values
df_clean = df[delay_columns].dropna()

#descriptive analytics
#correlation heatmap
plt.figure(figsize=(12,8))
corr_matrix = df_clean.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')      #generated with the help of AI
plt.title("Correlation Matrix of Flight Delay Related Variables")
plt.tight_layout()
plt.savefig("descriptive_correlation_heatmap.png")
plt.close()

#predictive analyticis
#select predictors based on correlation with ARR_DELAY
predictors = ['DEP_DELAY', 'CARRIER_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'NAS_DELAY']
X = df_clean[predictors]
y = df_clean['ARR_DELAY']

#add constant for intercception
x = sm.add_constant(X)    #generated with the help of AI

#fit linear regression model
model = sm.OLS(y, X).fit()

with open("predictive_model_summary.txt", "w") as f:  #generated with the help of AI
    f.write(str(model.summary()))    #generated witht he help of AI

#This analysis was conducted with assistance from ChatGPT, an AI language model developed by OpenAI, for help with Python coding, data interpretation, and report structuring.
