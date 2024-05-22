#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


df=pd.read_csv('C:/Users/Shubhangi/Desktop/shopping_trends.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


percentage_missingval=(df.isna().sum()*100/len(df)).round(2) 
percentage_missingval


# In[7]:


#filling null values
m=df["Age"].median()
df['Age'] = df['Age'].fillna(m)


# In[8]:


m1=df["Review Rating"].median()
df['Review Rating'] = df['Review Rating'].fillna(m1)


# In[9]:


m2=df["Previous Purchases"].median()
df['Previous Purchases'] = df['Previous Purchases'].fillna(m2)


# In[10]:


(df.isna().sum()*100/len(df)).round(2) 


# In[11]:


df.duplicated().sum()


# In[12]:


import numpy
def outliers(col_df):
  q1=np.percentile(col_df,25)
  q2=np.percentile(col_df,50)
  q3=np.percentile(col_df,75)
  iqr=q3-q1
  upper=q3+1.5*iqr
  lower=q1-1.5*iqr
  ol=col_df[(col_df>upper)|(col_df<lower)]
  return bool(len(ol))


# In[13]:


import numpy as np   #no outliers founded
num_col= df.select_dtypes([int,float])
for col in num_col:
  result= outliers(df[col])
  print(f'{col}:{result}')


# In[14]:


# whats the difference in previosuly purchased amount and present purchase amount?
b=df[['Previous Purchases','Purchase Amount (USD)']]
b.boxplot()
plt.xlabel('Target Columns')
plt.ylabel('Purchase value in dollars')
plt.title('Distribution of Variables')
plt.grid(alpha=0.3)
plt.show()


# In[15]:


#To check the  distributions of shipping methods opted by customers
shipping=df['Shipping Type'].value_counts()
plt.figure(figsize = (10,5))
plt.bar(shipping.index,shipping.values,color='orange',edgecolor='green')
plt.title("methods of shippings")
plt.xlabel("types of shippings")
plt.ylabel("total counts")
plt.show()


# In[16]:


#If &How the actual payment methods and preferred payment methods differ
pref1=df['Payment Method'].value_counts()
pref=df['Preferred Payment Method'].value_counts()
plt.figure(figsize = (10,5))
pref1.plot(color='r',label='Payment Method')
pref.plot(color='g',label='Prefferred Payment Method')
plt.ylabel("counts")
plt.xlabel("paymanet options")
plt.legend()
plt.show()


# In[17]:


#Which category has the highest rating in different seasons
sns.catplot(data=df, x="Season", y="Review Rating",col="Category",kind='bar',height=4, aspect=.9)
plt.show()


# In[18]:


#To check the hike increase or decrease in two consecutive purchases
df['Purchase Gap']=((df[['Previous Purchases','Purchase Amount (USD)']].
                     pct_change(axis=1)['Purchase Amount (USD)'])*100).round(2).map(str)+'%'
#Details of customers with highest and lowest hike in two consecutive purchase


# In[19]:


max(df['Purchase Gap'])


# In[20]:


row_index1 = df.index[df['Purchase Gap'] == '988.89%'].tolist()
print(row_index1)


# In[21]:


df.loc[[1496,2171,3394]]


# In[22]:


min(df['Purchase Gap'])


# In[23]:


row_index1 = df.index[df['Purchase Gap'] == '-10.0%'].tolist()
print(row_index1)


# In[24]:


df.loc[[2917, 2967, 3602]]


# In[25]:


#Which location is giving the best possible review 
df_rating=df.groupby('Location')['Review Rating'].mean().reset_index()
fig = px.line(df_rating, x='Location', y="Review Rating",markers=True)
fig.show()


# In[26]:


#The relationship between customers from specific location and their frequency of purchase
df_corr=df.pivot_table(columns='Location',index='Frequency of Purchases',aggfunc='size')
plt.figure(figsize = (20,7))
sns.heatmap(df_corr,fmt='d',annot=True,cmap='coolwarm')  #annot dsplays the data values
plt.show()


# In[27]:


#To check which age group spent the most in Purchase and its respective season
season_wiseitem=df.groupby(['Season','Age'])["Purchase Amount (USD)"].mean().reset_index()
#for summer season
s=season_wiseitem.loc[season_wiseitem['Season'] == 'Summer']
s.loc[s['Purchase Amount (USD)'].idxmax()]


# In[28]:


#for winter season
w=season_wiseitem.loc[season_wiseitem['Season'] == 'Winter']
w.loc[w['Purchase Amount (USD)'].idxmax()]


# In[29]:


#for fall 
f=season_wiseitem.loc[season_wiseitem['Season'] == 'Fall']
f.loc[f['Purchase Amount (USD)'].idxmax()]


# In[30]:


#for spring
sp=season_wiseitem.loc[season_wiseitem['Season'] == 'Spring']
sp.loc[sp['Purchase Amount (USD)'].idxmax()]


# In[31]:


#To check the avg purchase value by different age groups over all seasons
(px.bar(season_wiseitem, x='Age',y='Purchase Amount (USD)', color='Season',text_auto=True,
barmode='group',title='Average purchased value by age groups over the seasons')
.update_layout(title_font_size=20)
.update_xaxes(showgrid=True)
).show()


# In[32]:


#To get the detail  information about  each category with respect to review ratings
k=df.groupby("Category")[["Review Rating"]].aggregate([min,max,'mean'])
print("Minimum,Maximum and Average ratings for different categories purchased")
k


# In[33]:


#visualize the item purchased in all 4 seasons 
plt.figure(figsize = (20,7))
sns.histplot(df, x="Season", hue="Item Purchased", multiple="dodge",stat="count",shrink=.8)
plt.title("Item Purchased Season wise")
plt.ylabel("number of item purchased")
plt.show()


# In[34]:


#whats the probability of customer aged more and less than 40 years giving more than 4 star rating?
#Customer aged more than 40
Total_customers = df[df['Age']>40].shape[0]
more_than_4_review = df[df['Review Rating'] >4 ].shape[0]
probability_of_customers_giving_more_than_4_rating_old = (more_than_4_review/Total_customers)*100
print("probability of customers giving more than 4 star ratings older than 40 years old is :", 
      probability_of_customers_giving_more_than_4_rating_old )


# In[35]:


#customer aged less than 40
Total_customers = df[df['Age']<40].shape[0]
more_than_4_review = df[df['Review Rating'] >4 ].shape[0]
probability_of_customers_giving_more_than_4_rating_young = (more_than_4_review/Total_customers)*100
print("probability of customers giving more than 4 star ratings younger than 40 years old is :", 
      probability_of_customers_giving_more_than_4_rating_young)


# In[36]:


#To check if the genders are playing important role in opting for subscription?
plt.figure(figsize = (10,5))
sns.countplot(x = 'Subscription Status', data = df, hue = 'Gender', palette = 'dark')
plt.ylabel("Number of customers")
plt.title('Subscription Status vs Gender', fontweight = 30, fontsize = 20)
plt.show()


# In[37]:


#to get idea of the requirement of different sizes of different categories through various loactions
df_1=df.groupby(["Category","Location"])['Size'].value_counts().to_frame(name='count')
df_1
df_1.to_csv('C:/Users/Shubhangi/Desktop/df_1.csv')  


# In[38]:


df_1.to_csv('C:/Users/Shubhangi/Desktop/df_1.csv')  


# In[ ]:




