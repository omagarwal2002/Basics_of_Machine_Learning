#%%
import pandas as pd
#%%
x=pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
#%%
print(x)
#%%
y=pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
print(y)
#%%
#creating a series
z=pd.Series([1, 2, 3, 4, 5])
print(z)
#%%
print(y.iloc[0])#printing only row 1
#%%
print(y.iloc[:,0])#printing only column 1
#%%
print(y.iloc[:1,0])#printing only row1 of column 1
#%%
print(y.iloc[0:2,0])#printing only row0 and row1 of column 1
#%%
print(y.iloc[[0, 1], 0])#printing only row0 and row1 of column 0
#%%
print(y.iloc[-1:])#printing from last
#%%
print(x.loc[0, 'Yes'])#printing row 0 and column Yes element
#%%
print(x.loc[:, ['Yes','No']])
#%%
print(x.Yes==50)
#%%
print(x.loc[(x.Yes == 50) & (x.No >= 90)]) # can also use | other than &
#%%
print(x.loc[x.Yes.isin([50, 21])])