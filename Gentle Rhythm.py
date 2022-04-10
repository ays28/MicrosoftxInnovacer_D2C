import pyActigraphy as pac
#installing tool for reading Actigrapgy readings

import os
fpath = os.path.join(os.path.dirname(pac.__file__),'tests/data/')

#Using actigraphy data available with Pyactigraphy 
raw = pac.io.read_raw_awd(fpath+'example_01.AWD')
crespo_6h = raw.Crespo(alpha='6h')

df = raw.data.index.astype(str)
df1 = raw.data
df1

import matplotlib.pyplot as plt
x=df
y=df1
plt.plot(x,y)
plt.show()

import pandas as pd
df4 = pd.DataFrame(df1)

result = [group[1] for group in df4.groupby(df4.index.date)]
######################################################################
#Creating input from the Actigraphy data, we assume repetition of movement
#therefore we have only used 4 days of data which is 5760 time stamps to train
#the network
import numpy as np
a=result[1]
for i in range(len(a)):
  r = a[0][i]
  m=list()
  m.append(a)

a=result[2]
for i in range(len(a)):
  r = a[0][i]
  n=list()
  n.append(a)

a=result[3]
for i in range(len(a)):
  r = a[0][i]
  l=list()
  l.append(a)

a=result[4]
for i in range(len(a)):
  r = a[0][i]
  o=list()
  o.append(a) 
#########################################################
#due to limited data we have created similar activity pattern using four days of data
c = np.array([m,n,l,o,m,n,l,o,m,n,l,o])
c.shape
X = c
#########################################################

#using Crespo's Algorith to get rest duration, using rest duration to determin sleep quality 
#We've assigned 1 for Good Sleep quality and 0 for Bad Sleep Quality
y = list()
crespo_6h = raw.Crespo(alpha='6h')
aot = raw.Crespo_AoT()
delta = aot[0]-aot[1]
for i in range(12):
  #print (i)
  if delta[i]>= delta[5]:
    y.append([1])
  if delta[i] < delta[5]:
    y.append([0])
  
y =array(y)

##########################################################
# Creating Deep Learnig Network using LSTMs
model = Sequential()
model.add(LSTM(1440, activation='sigmoid', return_sequences=True, input_shape=(1,1440)))
model.add(LSTM(1440, activation='sigmoid'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# Traning the Model
model.fit(X, y, epochs=200, verbose=0)
# Prediction System

a=result[12]
for i in range(len(a/2)):
  r = a[0][i]
  k=list()
  k.append(a) 

x_input = array([k])
x_input = x_input.reshape((1, 1, 1440))
Sleep_health = model.predict(x_input, verbose=0)

if Sleep_health < 0.5:
  print ("Ongoing activities can result in Poor Sleep Quality, please refer to expert suggestions")
if Sleep_health >0.5:
  print ("All seems well, have a good day")

