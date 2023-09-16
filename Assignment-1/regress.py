print("----------Running the code for method-2 i.e regress method----------")

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


def make_prediction(X_seen, class_attribute_seen, Xtest, class_attribute_unseen, lambda_value):
    # Computing the matrix A_s (40x85) and M_s (40x4096)
    A_s = class_attribute_seen
    M_s=np.zeros([40,4096])
    for i in range(X_seen.shape[0]):
        M_s[i]=X_seen[i].mean(axis=0)

    # Computing matrix W using the formula and it ssould be (85x4096)
    first_term = np.linalg.inv(np.dot(A_s.T, A_s) +lambda_value* np.identity(A_s.shape[1]))
    next_term=np.dot(A_s.T, M_s)
    W = np.dot(first_term, next_term)

    #calculating means_unseen for unseen classes of (10x4096)
    means_unseen=np.dot(class_attribute_unseen, W)

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
    # print(f"Size of prediction array: {predictions.size}")
    return predictions

#Testing the accuracy for all the values of lambda
lambda_values = [0.01, 0.1, 1, 10, 20, 50, 100]
lambda_accuracy=[]
for lambda_value in lambda_values:
    predictions=make_prediction(X_seen, class_attribute_seen, Xtest, class_attribute_unseen, lambda_value)
    accuracy = accuracy_score(Ytest, predictions)
    lambda_accuracy.append(accuracy)
    print(f"Lambda = {lambda_value},  Accuracy: {accuracy * 100:.2f}%")
max_lamda_index=np.argmax(lambda_accuracy)
lambda_val_for_max_accuracy=lambda_values[max_lamda_index]
max_val_of_accuracy=lambda_accuracy[max_lamda_index]
print(f"maximum accuracy is: {max_val_of_accuracy* 100:.2f}% for lambda: {lambda_val_for_max_accuracy}")