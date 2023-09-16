print("----------Running the code for method-1 i.e convex method----------")

#importing libraries
import numpy as np
from sklearn.metrics import accuracy_score

#File paths (Change accordingly)
class_attributes_seen_path="data/AwA_python/class_attributes_seen.npy"
class_attributes_unseen_path="data/AwA_python/class_attributes_unseen.npy"
X_seen_path="data/AwA_python/X_seen.npy"
Xtest_path="data/AwA_python/Xtest.npy"
Ytest_path="data/AwA_python/Ytest.npy"

#Loading all the datasets given as npy file
X_seen=np.load(X_seen_path,encoding='bytes',allow_pickle=True)
class_attribute_seen=np.load(class_attributes_seen_path,encoding='bytes',allow_pickle=True)
class_attribute_unseen=np.load(class_attributes_unseen_path,encoding='bytes',allow_pickle=True)
Xtest=np.load(Xtest_path,encoding='bytes',allow_pickle=True)
Ytest=np.load(Ytest_path,encoding='bytes',allow_pickle=True)
# Conforming the shape given in readme file by printing after loading them
print("Class Attribute Seen Shape: ",class_attribute_seen.shape)
print("Class Attribute Unseen Shape: ",class_attribute_unseen.shape)
print("X_Seen Shape: ",X_seen.shape)
print("Xtest Shape: ",Xtest.shape)
print("Ytest Shape: ",Ytest.shape)


# Computing the mean of each seen class
means_seen=np.zeros([40,4096])
for i in range(X_seen.shape[0]):
    means_seen[i]=X_seen[i].mean(axis=0)
print("Size of mean for seen:", means_seen.shape)

# Computing the similarity (dot product based) of each unseen class with each of the seen classes.
similarities = np.dot(class_attribute_unseen, class_attribute_seen.T)
#Normalizing the similarity vector (to make it sum to 1)
similarities /= similarities.sum(axis=1, keepdims=True)
print(f"Shape of similarity matrix: {similarities.shape}")

# Computing the mean of each unseen class using a convex combination of means of seen classes
means_unseen = np.dot(similarities, means_seen)
print(f"Shape of unseen means: {means_unseen.shape}")

# Predicting the classes for the Xtest
len_Xtest = Xtest.shape[0]
num_unseen_classes = means_unseen.shape[0]

# Initializing the prediction array
predictions = np.empty(len_Xtest, dtype=float)

# Calculating the euclidean distance on x* with each unseen classes
for i in range(len_Xtest):
    temp = Xtest[i]
    distances = np.empty(num_unseen_classes)
    
    for class_index in range(num_unseen_classes):
        distance = np.linalg.norm(temp - means_unseen[class_index])
        distances [class_index] = distance
    
    # Assign the class with the minimum distance as the prediction for the temp
    predictions[i] = np.argmin(distances) +1 #adding 1 bcs it will give range between 0 to 9 but we need 1 to 10
print(f"Size of prediction array: {predictions.size}")


accuracy=accuracy_score(Ytest, predictions)
print(f"Accuracy for Method-1 is: {accuracy*100:.2f}%")