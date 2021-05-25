#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.figure_factory as ff
import pandas as pd
from plotly.offline import iplot
import plotly.graph_objects as go

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv")


# In[2]:


table_data = [['School','Women','Men','Gap'],
              ['MIT',94,152,58],
              ['Stanford',96,151,55],
              ['Harvard',112,165,53],
              ['U.Penn',92,141,49],
              ['Princeton',90,137,47],
              ['Chicago',78,118,40],
              ['Georgetown',94,131,37],
              ['Tufts',76,112,36],
              ['Yale',79,114,35],
              ['Columbia',86,119,33],
              ['Duke',93,124,31],
              ['Dartmouth',84,114,30],
              ['NYU',67,94,27],
              ['Notre Dame',73,100,27],
              ['Cornell',80,107,27],
              ['Michigan',62,84,22],
              ['Brown',72,92,20],
              ['Berkeley',71,88,17],
              ['Emory',68,82,14],
              ['UCLA',64,78,14],
              ['SoCal',72,81,9]]

fig = ff.create_table(table_data)

trace1 = go.Bar(x=df.School, y=df.Men, xaxis='x2', yaxis='y2', marker=dict(color='#0099ff'), name = 'Men earnings')
trace2 = go.Bar(x=df.School, y=df.Women, xaxis='x2', yaxis='y2', marker=dict(color='#404040'), name = 'Women earnings')
trace3 = go.Scatter(x=df.School, y=df.Gap, xaxis='x2', yaxis='y2', marker=dict(color='#ff0000'), name = 'Gap')
fig.add_traces([trace1, trace2, trace3])

fig['layout']['xaxis2'] = {}
fig['layout']['yaxis2'] = {}
                
fig.layout.yaxis.update({'domain': [0, .45]})
fig.layout.yaxis2.update({'domain': [.6, 1]})
                
fig.layout.yaxis2.update({'anchor': 'x2'})
fig.layout.xaxis2.update({'anchor': 'y2'})
fig.layout.yaxis2.update({'title': 'earnings'})
                
fig.layout.margin.update({'t':75, 'l':50})
fig.layout.update({'title': 'Gender Earning Gap by School'})

fig.layout.update({'height':800})

fig.show()
                   


# In[ ]:




