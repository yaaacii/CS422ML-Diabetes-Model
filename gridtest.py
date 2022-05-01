import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Generate dependent variable
y = data.iloc[:,0]
# Generate matrix of features
X = data.iloc[:,1:-1]

from sklearn import preprocessing
scaledX = preprocessing.normalize(X, axis=1)
X = pd.DataFrame(scaledX, columns=X.columns)

# print(X.shape, y.shape)

'''Splitting dataset into training and testing dataset''' 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

resultsfile = open("grid_results.csv", "w")
resultsfile.write("Accuracy,Optimizer,Loss_Function,Learning_Rate,Batch_Size,Node_Count,Epochs\n")
resultsfile.close()


# grid test

Optimizers = ['SGD', 'Adam']
LossFunctions = ['binary_crossentropy', 'mean_squared_error']
LearningRates = [0.01, 0.001, 0.0001, 0.00001]
BatchSizes = [32,64,128]
MaxEpochs = [100,200,300]
NodeCounts = [4,5,6,7,8]

for epochs in MaxEpochs:
    for loss_function in LossFunctions:
        for batch_size in BatchSizes:
            for learning_rate in LearningRates:
                for node_count in NodeCounts:
                    for optimizer in Optimizers:   
                        if optimizer == 'SGD':
                            optimizer_ = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                        elif optimizer == 'Adam':
                            optimizer_ =  tf.keras.optimizers.Adam(learning_rate=learning_rate)
                        
                        ann = tf.keras.models.Sequential()
                        ann.add(tf.keras.layers.Dense(units=node_count,activation="relu", input_shape = (20,)))
                        ann.add(tf.keras.layers.Dense(units=node_count,activation="relu"))
                        
                        ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
                        
                        '''Compiling ANN'''
                        ann.compile(optimizer=optimizer_, loss=loss_function, metrics=['accuracy'])
                        ann.fit(x=X_train,y=y_train,batch_size=32,epochs = epochs)
                        # , validation_data=[X_valid, y_valid]
                        
                        from sklearn.metrics import accuracy_score
                        y_pred = ann.predict(X_test)
                        class_predictions = [round(num[0]) for num in y_pred]                    
                        
                        accuracy = accuracy_score(y_test, class_predictions)
                        results_string = str(accuracy) + "," + optimizer + "," + loss_function + "," + str(learning_rate) + "," + str(batch_size) + "," + str(node_count) + "," + str(epochs) + "\n"
                        print(results_string)

                        #  # Evaluating the Algorithm 
                        # from sklearn.metrics import classification_report, confusion_matrix
                        # print(confusion_matrix(y_test, y_pred))
                        # print(classification_report(y_test, y_pred))
                                                
                        resultsfile = open("grid_results.csv", "a")
                        resultsfile.write(results_string)
                        resultsfile.close()
                    
                    
                    
                    
                    
                    
                    