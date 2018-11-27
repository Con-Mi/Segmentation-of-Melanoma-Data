from os import listdir
import pandas as pd

def get_id_csv_files():
    # Training data
    list_ids_train_input = [f for f in listdir("./MelanomaData/ISIC2018_Task1-2_Training_Input")]
    list_ids_train_labels = [f for f in listdir("./MelanomaData/ISIC2018_Task1_Training_GroundTruth")]

    list_ids_train_input = sorted(list_ids_train_input)
    list_ids_train_labels = sorted(list_ids_train_labels)

    dict_train_inputs = {"ids": list_ids_train_input}
    dict_train_labels = {"ids": list_ids_train_labels}

    train_input_df = pd.DataFrame(dict_train_inputs, index = None)
    train_labels_df = pd.DataFrame(dict_train_labels, index = None)
    train_input_df.to_csv("./MelanomaData/train_input_ids.csv")
    train_labels_df.to_csv("./MelanomaData/train_labels_ids.csv")

    # Test Validation Data
    list_ids_valid_input = [f for f in listdir("./MelanomaData/ISIC2018_Task1-2_Validation_Input")]

    list_ids_valid_input = sorted(list_ids_valid_input)

    dict_valid_inputs = {"ids": list_ids_valid_input}

    valid_input_df = pd.DataFrame(dict_valid_inputs, index = None)
    valid_input_df.to_csv("./MelanomaData/valid_input_ids.csv")

    # Test Data
    list_ids_test_input = [f for f in listdir("./MelanomaData/ISIC2018_Task1-2_Test_Input")]

    list_ids_test_input = sorted(list_ids_test_input)

    dict_test_inputs = {"ids": list_ids_test_input}

    test_input_df = pd.DataFrame(dict_test_inputs, index = None)
    test_input_df.to_csv("./MelanomaData/test_input_ids.csv")

get_id_csv_files()