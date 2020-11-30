import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from IPython.display import display, HTML

data = pd.read_csv(r'C:\Users\admin\Desktop\E_commerce_data.csv', engine='python')
# print(data.head())
print(data.InvoiceNo.str.contains('C').sum())
print(data.InvoiceNo.isna().sum())

# sns.boxplot(data.UnitPrice)
# plt.show()
# sns.boxplot(data.InvoiceNo)
# plt.show()
# print(data.InvoiceNo.dtype)
def clean_InvoiceNo(InvoiceNo):
    if InvoiceNo[0] == 'C':
        return InvoiceNo.replace(InvoiceNo[0], '')
    else:
        return InvoiceNo
data['InvoiceNo'] = data['InvoiceNo'].apply(clean_InvoiceNo)
# print(data.shape)
# print(data.info())
# print(data.describe())
data.Description.fillna(data.Description.mode().values[0], inplace=True)
data.CustomerID.fillna('Unknown', inplace=True)
data['Sales'] = data.UnitPrice * data.Quantity
data.InvoiceDate = pd.to_datetime(data.InvoiceDate)
data['Month'] = data.InvoiceDate.dt.month
data['Hour'] = data.InvoiceDate.dt.hour
data['DayOfWeek'] = data.InvoiceDate.dt.dayofweek
data['DayOfWeek'] = data['DayOfWeek'].map({0: '1_Mon', 1: '2_Tues', 2: '3_Wed', 3: '4_Thur', 4: '5_Fri', 5: '6_Sat',
                                           6: '7_Sun'})

# print(data.columns)
# print(data.columns.dtype)
# print(data.Sales)
# print(data.isna().sum())
# print(data.Description.mode()[0])

data = data[(data.Quantity < 15000) & (data.Quantity > -15000)]
data = data[(data.UnitPrice >= 0) & (data.UnitPrice < 20000)]
# print(data.Quantity.describe())
# print(data.UnitPrice.describe())


# 1. Which Country has highest Sales

result = data.groupby('Country').sum()
result.sort_values(by=['Sales'], inplace=True, ascending=False)
print(result)
#
countries = [country for country, df in data.groupby('Country')]
# countries = [data.Country.unique()]
# # print((countries))
plot = plt.bar(countries, result['Sales'])
plt.xticks(rotation=90)
plt.yticks(result['Sales'])
plt.show()

# 2. Which costumer has highest sales
result = data.groupby('CustomerID').sum()[['Sales', 'Quantity']].reset_index()
# print(result.sort_values(['Sales', 'Quantity'], ascending=False))
# result = data.groupby('CustomerID').sum()
# result.sort_values(by=['Sales', 'Quantity'], ascending=False, inplace=True)
# print(result)

Revenue = data.groupby('CustomerID')[['Sales', 'Quantity']].agg(['sum', 'mean', 'median'])
# Revenue.sort_values(by=[('Sales','sum')], inplace=True, ascending=False)
# # Revenue.sort_values(by=['Sales', 'Quantity'], inplace=True, ascending=False)
# # print(Revenue[1:].reset_index())
# print(Revenue.iloc[1:][('Sales', 'sum')].describe())
# print(Revenue.iloc[1:][['Sales', 'Quantity']].describe())

# Now lets plot a graph
# sns.distplot(Revenue.iloc[1:][('Sales', 'sum')], kde=False)
# plt.show()

# 3. which customerID returns the order most
data = data[data.Quantity < 0]
# Item_Returned = data.groupby('CustomerID')[['Sales', 'Quantity']].sum()
# Item_Returned.sort_values(by=['Quantity'], inplace=True, ascending=True)
# Item_Returned = data.groupby('CustomerID')[['Sales', 'Quantity']].agg(['sum'])
# Item_Returned.sort_values(by=[('Quantity', 'sum')], inplace=True)
# Item_Returned = Item_Returned.head(10)
# print(Item_Returned)

# sns.barplot(x =Item_Returned.index, y = abs(Item_Returned['Quantity']))
# sns.barplot(x =Item_Returned.index, y = abs(Item_Returned[('Quantity', 'sum')]))
# plt.ylabel('No. of Items Returned')
# plt.xticks(rotation=90)
# plt.show()

# 4. Which Product is Sold Most And Least

Item_sold_most = data.groupby('Description')[['Sales', 'Quantity']].sum()
# Item_sold_most.sort_values(by=['Quantity'], inplace=True, ascending=False)
# Item_sold_most = Item_sold_most.head(10)
# print(Item_sold_most)

# Item_sold_most = data.groupby('StockCode')[['Sales', 'Quantity']].sum()
# Based on UnitPrice calculate the Quantity of Stockcode
# Item_sold_most = data.groupby(['StockCode', 'UnitPrice'])[['Quantity']].sum()
# Item_sold_most.sort_values(by='Quantity', inplace=True, ascending=False)
# Item_sold_most = Item_sold_most.head(10)
# print(Item_sold_most)

# Item_sold_most1 = data.groupby('StockCode')[['Quantity']].sum()
# Item_sold_most1.sort_values(by='Quantity', inplace=True, ascending=False)
# Item_sold_most1 = Item_sold_most1.head(10)

# Item_sold_most1 = data.groupby('StockCode').sum()
# Item_sold_most1.sort_values(by='Quantity', inplace=True, ascending=False)
# Item_sold_most1 = Item_sold_most1.Quantity.head(10)
# print(Item_sold_most1)

# plt.figure(figsize=(8,6))
# sns.barplot(x=Item_sold_most1.index, y=Item_sold_most1.Quantity)
# plt.ylabel('Quantity of Product Returned')
# plt.xlabel('StockCodes')
# plt.show()

# plt.figure(figsize=(9, 7))
# sns.barplot(x=Item_sold_most.index, y=Item_sold_most['Quantity'])
# plt.ylabel('Total no. of product sold')
# plt.xticks(rotation=60, fontsize=6)
# plt.show()

# 5. Which Product is Preferred least

Item_sold_most1 = data.groupby('StockCode')[['Quantity']].sum()
# Item_sold_most1.sort_values(by='Quantity', inplace=True)
# Item_sold_most1 = Item_sold_most1[Item_sold_most1['Quantity'] == 0]
# print('Items preferred least are', len(Item_sold_most1))
# print(Item_sold_most1)

# 6. Which Country bring more revenue in total and average ?
#
Most_Revenue_Country = data.groupby('Country')[['Sales']].sum()
# Most_Revenue_Country.sort_values(by='Sales', inplace=True, ascending=False)
# # Most_Revenue_Country = Most_Revenue_Country.head(10)
# print(Most_Revenue_Country)
#
# plt.figure(constrained_layout=True, figsize=(8,6))
# sns.barplot(x=Most_Revenue_Country.index, y=Most_Revenue_Country.Sales)
# plt.xlabel('Countries')
# plt.ylabel('Total Amount of Sales/Revenue')
# plt.xticks(rotation=70)
# plt.show()

# Lets draw a graph without the United Kingdom
# Most_Revenue_Country.drop('United Kingdom', inplace=True)
# plt.figure(constrained_layout=True, figsize=(8,6))
# # sns.barplot(x=Most_Revenue_Country.index, y=Most_Revenue_Country.Sales)
# sns.barplot(y=Most_Revenue_Country.index, x=Most_Revenue_Country.Sales)
# plt.xlabel('Countries')
# plt.ylabel('Total Amount of Sales/Revenue')
# # plt.xticks(rotation=70)
# plt.show()

# print(data.InvoiceNo.dtype)
# Invoice_Country = data.groupby('Country')[['InvoiceNo']].count()
# Invoice_Country.sort_values(by='InvoiceNo', inplace=True, ascending=False)
# Invoice_Country = Invoice_Country.head(10)
# print(Invoice_Country)

# lets draw the graph

# sns.barplot(x=Invoice_Country.index, y=Invoice_Country.InvoiceNo.astype(float))
# plt.xlabel('Countries')
# plt.ylabel('Total no. of Orders')
# plt.xticks(rotation=65)
# plt.yticks([1000, 2000, 8000, 9000, 100000, 300000, 500000])
# plt.show()


# Plotly Graph

# fig = go.Figure(data=go.Choropleth(locations=Invoice_Country.index,
#                                    z=Invoice_Country.InvoiceNo.astype(float),
#                                    locationmode = 'country names',
#                                    colorscale = 'Greens',
#                                    colorbar_title='Order number'))
#
# fig.update_layout(title_text = 'Order number per country',
#                   geo = dict(showframe=True, projection={'type': 'mercator'}))
# fig.layout.template = None
# fig.show()

# 8. Lets find the Countries Qauntity Numbers
# Most_Revenue_Country = data.groupby('Country')[['Quantity']].sum()
# Most_Revenue_Country.sort_values(by='Quantity', inplace=True, ascending=False)
# Most_Revenue_Country = Most_Revenue_Country.head(10)
# print(Most_Revenue_Country)

# We Can Plot the Graphs With and Without United Kingdom

# Now Lets find the Countries Quanttiy of Products Purchased based on the UnitPrice
# Most_Revenue_Country = data.groupby('Country')[['UnitPrice']].agg(['sum', 'mean'])
# Most_Revenue_Country.sort_values(by=[('UnitPrice', 'mean')], inplace=True, ascending=False)
# Most_Revenue_Country = Most_Revenue_Country.head(10)
# print(Most_Revenue_Country)
# Here we are using mean becoz the total sum of high unitprice product can we max, as compared to more quantiy of products
# less UnitPrice so here we are using the mean for the unit price cal. to analyze the total no. product per unit price are
# purchased by each country

# plt.figure(constrained_layout=True, figsize=(8,6))
# sns.barplot(y=Most_Revenue_Country.index, x=Most_Revenue_Country[('UnitPrice', 'mean')])
# plt.ylabel('Countries')
# plt.xlabel('Mean values of total no. of products bought per total no. of UnitPrices')
# # plt.xticks(rotation=90)
# plt.show()

# 7. Which month we sell out most and least ?
# Most_Month_Sales = data.groupby('Month')['Sales'].agg(['sum', 'mean'])
# Most_Month_Sales.sort_values(by=[('Sales', 'sum')], inplace=True, ascending=False)
# Most_Month_Sales =Most_Month_Sales.head(10)
# print(Most_Month_Sales)

# Here we can observe that the max sale increases in November drastically but the average sales is almost same for entire year

# Lets draw the Graph for mean and sum of sales as per month

# fig, axes = plt.subplots(1, 2, figsize=(12,6))
# axes = axes.flatten()
#
# sns.barplot(x=Most_Month_Sales.index, y=Most_Month_Sales['sum'], ax=axes[0])#.set_title('Total Revenue over a year')
# plt.title('Total Revenue over a year')
# plt.ylabel('a')
#
# sns.barplot(x=Most_Month_Sales.index, y=Most_Month_Sales['mean'], ax=axes[1])#.set_title('Average Revenue over a year')
# plt.title('Average Revenue over a year')
# plt.show()

# 8. What time do people tend to buy products more ?
# Most_product_bought_time = data.groupby(['Hour'])['Sales'].agg(['sum', 'mean'])
# Most_product_bought_time.sort_values(by='Quantity', inplace=True, ascending=False)
# print(Most_product_bought_time.head(10))

# Lets draw the graph
# fig, axes = plt.subplots(1, 2, figsize=(10, 6))
# axes = axes.flatten()

# sns.barplot(x=Most_product_bought_time.index, y=Most_product_bought_time['sum'], ax=axes[0]).set_title('Total sales over a day')

# sns.barplot(x=Most_product_bought_time.index, y=Most_product_bought_time['mean'], ax=axes[1]).set_title('Average sales over a day')

# plt.show()

# 9. Which day of week people tend to visit and purchase stuff ?
# Most_Day_Sales = data.groupby(['DayOfWeek'])['Sales'].agg(['sum', 'mean'])
#
# print(Most_Day_Sales)
#
# Lets draw the graph
# fig, axes = plt.subplots(1, 2, figsize=(10, 6))
# axes = axes.flatten()
#
# sns.barplot(x=Most_Day_Sales.index, y=Most_Day_Sales['sum'], ax=axes[0]).set_title('Total sales over the days of a week')

# sns.barplot(x=Most_Day_Sales.index, y=Most_Day_Sales['mean'], ax=axes[1]).set_title('Total Average sales over the days of a week')

# plt.show()

# 10. Are there any relationship between repeat customers and all customer over the year ?

# First get the date range from the data
# print('Date Range: %s to %s' % (data['InvoiceDate'].min(), data['InvoiceDate'].max()))
# print(f"Date Range: {data['InvoiceDate'].min()} to {data['InvoiceDate'].max()}")
# print("Date Range: {0} to {1}".format(data['InvoiceDate'].min(), data['InvoiceDate'].max()))
# print('Date range:' + str(data['InvoiceDate'].min())+' ' + 'to'+ ' ' + str(data['InvoiceDate'].max()))

# data = data[data['InvoiceDate'] < '2011-12-01']

# Get total amount spent per invoice and associate it with CustomerID and Country
# Invoice_Customer_Sales = data.groupby(['InvoiceNo', 'InvoiceDate']).agg({'Sales': sum,
#                                                 'CustomerID': max, 'Country': max}).reset_index()
# print(Invoice_Customer_Sales)

# 11. Lets find the repeat customers monthly

# Monthly_Repeat_Customers = Invoice_Customer_Sales.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'), 'CustomerID']
#                                         ).filter(lambda x: len(x) > 1).resample('M').nunique()['CustomerID']
# print(Monthly_Repeat_Customers)

# Monthly Unique Customers

# Monthly_Unique_Customers = Invoice_Customer_Sales.set_index('InvoiceDate')['CustomerID'].resample('M').nunique()
# print(Monthly_Unique_Customers)

# Now Lets find the Ratio of Repeat to Unique Customer
# Monthly_Repeat_Percentage = Monthly_Repeat_Customers/Monthly_Unique_Customers*100.0
# print(Monthly_Repeat_Percentage)

# Now lets draw the graphs

# fig = plt.figure(constrained_layout=True, figsize=(8, 6))
# grid = gridspec.GridSpec(nrows=1, ncols=1,  figure=fig)
#
# ax = fig.add_subplot(grid[0, 0])
#
# pd.DataFrame(Monthly_Repeat_Customers.values).plot(ax=ax, figsize=(8,6))
#
# pd.DataFrame(Monthly_Unique_Customers.values).plot(ax=ax,grid=True)

# ax.set_xlabel('Date')
# ax.set_ylabel('Number of Customers')
# ax.set_title('Number of Unique vs. Repeat Customers Over Time')

# plt.xticks(range(len(Monthly_Repeat_Customers.index)), [x.strftime('%m.%Y') for x in Monthly_Repeat_Customers.index], rotation=45)
# ax.legend(['Repeat Customers', 'All Customers'])
# plt.show()

# 12. Let's investigate the relationship between revenue and repeat customers

# Basically here we are finding the total sales for each month
# Monthly_Sales = data.set_index('InvoiceDate')['Sales'].resample('M').sum()
# print(Monthly_Sales)
# Monthly_Sales.sort_values(by='Sales', inplace=True, ascending=False)
# print(Monthly_Sales)

# Monthly_Sales_Repeat_Customers = Invoice_Customer_Sales.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),
#                                     'CustomerID']).filter(lambda x: len(x)>1).resample('M').sum()['Sales']
# print(Monthly_Sales_Repeat_Customers)

# Let's get a percentage of the revenue from repeat customers to the overall monthly revenue

# Monthly_Sales_Percenatge_Repeat_Customers = Monthly_Sales_Repeat_Customers/Monthly_Sales * 100.0
# print(Monthly_Sales_Percenatge_Repeat_Customers)

# lets draw the graph

# fig = plt.figure(constrained_layout=True, figsize=(8,6))
# grid = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
#
# ax = fig.add_subplot(grid[0,0])
#
# pd.DataFrame(Monthly_Sales_Repeat_Customers.values).plot(ax=ax, figsize=(8,6))
#
# pd.DataFrame(Monthly_Sales.values).plot(ax=ax, grid=True)

# ax.set_xlabel('Date')
# ax.set_ylabel('Number of Customers')
# ax.set_title('Number of Unique vs. Repeat Customers Over Time')
# plt.xticks(range(len(Monthly_Repeat_Customers.index)), [x.strftime('%m.%Y') for x in Monthly_Repeat_Customers.index], rotation=45)
# ax.legend(['Repeat Customers', 'All Customers'])
# plt.show()

# 13. What are the items trend ?

# Now let's get quantity of each item sold per month
# Quantity_Item_Sold_PerMonth = data.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'), 'StockCode'])[['Quantity']].sum()
# Quantity_Item_Sold_PerMonth.sort_values(by='Quantity', inplace=True, ascending=False)
# print(Quantity_Item_Sold_PerMonth.head(15))

# Rank items by the last month's sales
# Last_Month_Sorted_Data = Quantity_Item_Sold_PerMonth.loc['2011-11-30'].reset_index()
# Last_Month_Sorted_Data.sort_values(by='Quantity', inplace=True, ascending=False)
# print(Last_Month_Sorted_Data.head(10))

# 14. Let's look at the top 5 items sale over a year

# Most_Sold_Item_OverYear = data.groupby(['StockCode', 'Description', 'UnitPrice'])[['Sales']].sum()
# Most_Sold_Item_OverYear.sort_values(by='Sales', inplace=True, ascending=False)
# print(Most_Sold_Item_OverYear.head(5))

# this will tell us about the whole year means which products leads in term of sales
# Most_Sold_Item_OverYear = data.groupby(['StockCode', 'Description'])[['Quantity']].sum()
# Most_Sold_Item_OverYear.sort_values(by='Quantity', inplace=True, ascending=False)
# print(Most_Sold_Item_OverYear.head(5))

# Most_Sold_Item_EachMonth = data.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'), 'StockCode',
#                                         'Description'])[['Quantity']].sum()
# Most_Sold_Item_EachMonth.sort_values(by='Quantity', inplace=True, ascending=False)
# print(Most_Sold_Item_EachMonth.head(5))

# 15. Lets look at as per the months which 5 products have topped the list
# Quantity_Item_Sold_PerMonth= data.loc[data['StockCode'].isin(['23084', '84826', '22197', '22086','85099B'])
# ].set_index('InvoiceDate').groupby([pd.Grouper(freq='M'),'StockCode','Description'])['Quantity'].sum().reset_index()
# print(Quantity_Item_Sold_PerMonth)

# Lets draw the graph
# Quantity_Item_Sold_PerMonth = Quantity_Item_Sold_PerMonth.reset_index()
#
# sns.set(style='whitegrid')
# plt.figure(constrained_layout=True, figsize=(8, 6))
# sns.lineplot(x=Quantity_Item_Sold_PerMonth['InvoiceDate'], y=Quantity_Item_Sold_PerMonth['Quantity'],
#              hue=Quantity_Item_Sold_PerMonth['StockCode'])
# plt.show()

# 16. Top 10 ReOrdered Items
# Reordered_Items = data.groupby(['StockCode', 'Description'])['InvoiceNo'].count().sort_values(ascending=False)
# print(Reordered_Items.head(10))

# 17. What is the Mall's Cancellation Rate ?
# Number_Canceled_Orders = data[data['Quantity']<0]['InvoiceNo'].nunique()
# print(Number_Canceled_Orders)
# Total_Orders = data['InvoiceNo'].nunique()
# print(Total_Orders)
# print("Mall's Cancellation Rate: {:.2f}%".format(Number_Canceled_Orders/Total_Orders * 100.0))

# 18. The revenue comes from repeat items or 1 items per month?
# Sales_Repeat_Items = data.groupby(['StockCode', 'Description'])['Sales'].sum().sort_values(ascending=False)
# Sales_Repeat_Items = data.groupby(['StockCode'])['Sales'].sum().sort_values(ascending=False)
# print(Sales_Repeat_Items.head(10))

# Monthly_Reorder_Items_Sales = data.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'), 'StockCode']).filter(lambda x: len(x) > 1).resample('M').sum()[['Sales']]
# Monthly_Reorder_Items_Sales.sort_values(by='Sales', ascending=False, inplace=True)
# print(Monthly_Reorder_Items_Sales)

# Monthly_One_Items_Sales = data.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'), 'StockCode']).filter(lambda x: len(x) == 1).resample('M').sum()['Sales']
# print(Monthly_One_Items_Sales)

# Monthly_Sales = data.set_index('InvoiceDate').groupby([pd.Grouper(freq='M')]).sum()['Sales']
# Monthly_Sales = data.set_index('InvoiceDate').groupby([pd.Grouper(freq='M')]).sum()['Sales'].sort_values(ascending=False)
# print(Monthly_Sales)

# Lets draw the graph
# fig = plt.figure(constrained_layout=True, figsize=(8, 6))
#
# ax = fig.add_subplot()
# pd.DataFrame(Monthly_Reorder_Items_Sales.values).plot(ax=ax, figsize=(12,8))
# pd.DataFrame(Monthly_Sales.values).plot(ax=ax,grid=True)
# pd.DataFrame(Monthly_One_Items_Sales.values).plot(ax=ax,grid=True)
#
# ax.set_xlabel('Date')
# ax.set_ylabel('Number of Customers')
# ax.set_title('Number of Unique vs. Repeat vs Total Items Over Time')
# plt.xticks(range(len(Monthly_Reorder_Items_Sales.index)), [x.strftime('%m.%Y') for x in Monthly_Reorder_Items_Sales.index], rotation=45)
# ax.legend(['Reordered Items', 'All Ordered Items', 'One Time Ordered Items'])
# plt.show()

# MODELLING
# RandomForest Regression
# Invoice_count = data.groupby(by='CustomerID', as_index=False)['InvoiceNo'].count()
# Invoice_ct = data.groupby(by='CustomerID', as_index=False)['Quantity'].count() # They are also same
# Invoice_ct = data.groupby('CustomerID', as_index=False)['InvoiceNo'].count() # will display same results
# Invoice_ct = data.groupby('CustomerID')['InvoiceNo'].count()
# Invoice_count.columns = ['CustomerID', 'NumberOrders']
# print(Invoice_ct)

# 19. This is the Average UnitPrice For Each at which they bought the products
# UnitPrice = data.groupby(by='CustomerID', as_index=False)['UnitPrice'].mean()
# UnitPrice.columns = ['CustomerID', 'Unitprice']
# print(UnitPrice)

# 20. Sales Made by each customer
# Sales = data.groupby(by='CustomerID', as_index=False)['Sales'].sum()
# Sales.columns=['CustomerID', 'Sales']
# Sales.sort_values(by='Sales', inplace=True, ascending=False)
# print(Sales)

# Total_items = data.groupby(by='CustomerID', as_index=False)['Quantity'].sum()
# # total_items.sort_values(by='Quantity', inplace=True, ascending=False)
# Total_items.columns = ['CustomerID', 'NumberItems']
# print(total_items)

# 21. Here We are finding the number of days between the customer first order/invoice(min) to now(InvoiceNo.max())

# Earliest_order = data.groupby(by='CustomerID', as_index=False)['InvoiceDate'].min()
# Earliest_order.columns = ['CustomerID', 'EarliestInvoice']
# Earliest_order['now'] = pd.to_datetime(data['InvoiceDate'].max())
# Earliest_order['now'] = data['InvoiceDate'].max()
# print(Earliest_order)

# Earliest_order['days_as_customer'] = 1 + (Earliest_order['now']-Earliest_order['EarliestInvoice']).dt.days
# Earliest_order.drop('now', axis=1, inplace=True)
# print(Earliest_order)

# Last_order = data.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
# Last_order.columns = ['CustomerID', 'last_purchase']
# Last_order['now'] = data['InvoiceDate'].max()
# Last_order['days_since_last_purchase'] = 1 + (Last_order.now - Last_order.last_purchase).astype('timedelta64[D]')
# Last_order['days_since_last_purchase'] = 1 + (Last_order.now - Last_order.last_purchase).dt.days
# Last_order.drop('now', axis=1, inplace=True)
# print(Last_order)

# Combine all the DataFrames into one
import functools

# DataFrames = [Invoice_count, UnitPrice, Sales, Earliest_order, Last_order, Total_items]
# CustomerTable = functools.reduce(lambda left, right: pd.merge(left, right, on='CustomerID', how='outer'), DataFrames)
# CustomerTable['OrderFrequency'] = CustomerTable['NumberOrders']/CustomerTable['days_as_customer']
# print(CustomerTable)

# print(CustomerTable.corr()['Sales'].sort_values(ascending = False))

# X = CustomerTable[['NumberOrders','Unitprice', 'days_as_customer', 'days_since_last_purchase', 'NumberItems', 'OrderFrequency']]
# y = CustomerTable['Revenue']

# Observation:
# NumberOrders and UnitPrice are 2 most important factors of forming revenue.
# days-as-customers and days-since-purcharse may not contribute much to see if a customer is loyal and bring most revenue to us.

