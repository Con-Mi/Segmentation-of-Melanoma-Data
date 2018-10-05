from os import listdir
import pandas as pd

list_ids_train_input = [f for f in listdir("./MelanomaData/ISIC2018_Task1-2_Training_Input")]
list_ids_train_labels = [f for f in listdir("./MelanomaData/ISIC2018_Task1_Training_GroundTruth")]

list_ids_train_input = sorted(list_ids_train_input)
list_ids_train_labels = sorted(list_ids_train_labels)

dict_train_inputs = {"ids": list_ids_train_input}
dict_train_labels = {"ids": list_ids_train_labels}

a = list_ids_train_input
b = list_ids_train_labels

train_input_df = pd.DataFrame(dict_train_inputs, index = None)
train_labels_df = pd.DataFrame(dict_train_labels, index = None)
train_input_df.to_csv("./MelanomaData/train_input_ids.csv")
train_labels_df.to_csv("./MelanomaData/train_labels_ids.csv")