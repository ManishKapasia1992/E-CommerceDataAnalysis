import pandas as pd
import os
import matplotlib.pyplot as plt


# Merging 12 months of data into a single CSV file

df = pd.read_csv("C://Users//admin//Desktop//Pandas-Data-Science-Tasks-master//SalesAnalysis//Sales_Data//Sales_April_2019.csv")
#print(df.head())
files = [file for file in os.listdir('C:\\Users\\admin\\Desktop\\Pandas-Data-Science-Tasks-master\\SalesAnalysis\\Sales_Data')]

all_months_data = pd.DataFrame()

for file in files:
    df = pd.read_csv("C://Users//admin//Desktop//Pandas-Data-Science-Tasks-master//SalesAnalysis//Sales_Data//"+file)
    all_months_data = pd.concat([all_months_data, df])

all_months_data.to_csv("C://Users//admin//Desktop//Pandas-Data-Science-Tasks-master//SalesAnalysis//all_data.csv", index=False)
all_data = pd.read_csv("C://Users//admin//Desktop//Pandas-Data-Science-Tasks-master//SalesAnalysis//all_data.csv")
#print(all_data.head())

# Clean the data
# Drop rows of NaN
nan_df = all_data[all_data.isna().any(axis=1)]
all_data = all_data.dropna(how='all')
#print(nan_df.head())
#print(all_data.head())

# To remove the data containing 'or' in order date

all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']
#print(all_data.head())

#all_data['month'] = 2
#all_data['month'] = all_data['Order Date'].iloc[0][0:2]
all_data['month'] = all_data['Order Date'].str[0:2]
all_data['month'] = all_data['month'].astype('int32')# to convert a string into integer
#print(all_data.head())

#print(all_data.isna().any(axis=1))
#print(all_data.head())

# Convert the columns Quantity Ordered and the Price Each into numeric values

all_data = all_data[all_data['Quantity Ordered'].str[0] != 'Q']
#all_data['Quantity Ordered']= all_data['Quantity Ordered'].str[0]
#all_data['Quantity Ordered']= all_data['Quantity Ordered'].astype('int32')
all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])
#print(all_data.head())


# Add the Sales column

all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
#print(all_data.head())

# Add the city column

def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

#all_data['City'] = all_data['Purchase Address'].apply(lambda x: x.split(',')[1])+ ' ' +all_data['Purchase Address'].str[2]
#all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + ' (' +get_state(x)+')')
all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")

#print(all_data.head())

# What was the best month for sales? How much was earned that month?

#results = all_data.groupby('month').sum()
#print(all_data.groupby('month').sum())

# Plot the graph of sales vs months

month = range(1,13)
#plt.bar(month, results['Sales'])
#plt.xticks(month)
#plt.xlabel('Sum in dollars')
#plt.ylabel('Months')
#plt.show()

# Which city had the highest no. of sales

result2 = all_data.groupby('City').sum()
#print(result2)
#cities = all_data['City'].unique() # This will unordered the cities values
cities = [city for city, df in all_data.groupby('City')]
#plt.bar(cities, result2['Sales'])
#plt.xticks(cities, rotation='vertical', size=8)
#plt.xlabel('Cities')
#plt.ylabel('Sales in Billions')
#plt.show()

# What is the best time to display adds to maximize likelihood of customer's buying product

all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
#print(all_data.groupby('Order Date').sum())
all_data['Hour'] = all_data['Order Date'].dt.hour
#all_data['Minute'] = all_data['Order Date'].dt.minute
#print(all_data.head())

hours = [hour for hour, df in all_data.groupby('Hour')]
results3 = all_data.groupby(['Hour']).count()
#print(results3)
'''plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel('Hours')
plt.ylabel('Number of Orders')
plt.grid()
plt.show()'''

# What products are most often sold together

#print(all_data.head())
df = all_data[all_data['Order ID'].duplicated(keep=False)]
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df = df[['Order ID', 'Grouped']].drop_duplicates()
#print(df.head(25))
from itertools import combinations
from collections import Counter

count = Counter()
for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))
for key, value in count.most_common(10):
    print(value, key)



# Which product sold most and why

#print(all_data.head())
product_group = all_data.groupby('Product')
#print(product_group.sum())
quantity_ordered = product_group.sum()['Quantity Ordered']

products = [product for product, df in product_group]

'''plt.bar(products, quantity_ordered)
plt.xticks(products, rotation='vertical', size=6)
plt.xlabel('Products')
plt.ylabel('No. Of Products Ordered')
#plt.show()'''

'''prices = all_data.groupby('Product').mean()['Price Each']
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticklabels(products, rotation='vertical', size=6)

plt.show()'''
