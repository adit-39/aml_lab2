import numpy as np

# Preprocessing the Data
with open("Mobile_Data.csv","r") as f:
	l = f.readlines()

X_train=[]
Y_train=[]

for row in l[1:]:
	curr = row.split(",")
	vec = curr[2:8]
	vec.insert(0,1)
	# converting all numeric values to floats
	temp = list(map(float,vec))
	X_train.append(temp)
	Y_train.append(float(curr[8].strip()))

# Saving part of the data for testing
X_test = np.mat(X_train[60:])
Y_test = Y_train[60:]

# Appplying Normal Equation for Linear Regression
X = np.mat(X_train[:60])
Y = np.mat(Y_train[:60])
Y = np.transpose(Y)
X_dag = np.linalg.pinv(X)
theta = X_dag * Y

# Using the theta vector to test 
Y_pred = X_test * theta

Err=[]
for i in range(len(Y_pred)):
	Y_p=float(Y_pred[i])
	error = ((Y_test[i]-Y_p)/Y_test[i])
	Err.append(error)

# Calculating error
Squares = list(map(lambda x: x*x,Err))
s_sq = sum(Squares)
m_sq = s_sq / len(Squares)
print "Percentage Error:",(m_sq ** 0.5)*100,"%"


