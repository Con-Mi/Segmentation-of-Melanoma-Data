#!bin/bash
YELLOW='\033[1;33m'
CYAN = '\033[0;36m'
NC='\033[0m'


mkdir MelanomaData
cd MelanomaData

# Training Set
echo -e "${YELLOW} Downloading Data"
echo -e "${NC}"
wget --output-file=train_data https://challenge.kitware.com/api/v1/item/5ac37a9d56357d4ff856e176/download

# Training Set Labels
echo -e "${YELLOW} Downloading Ground Truth Data"
echo -e "${NC}"
wget --output-file=train_data_truth https://challenge.kitware.com/api/v1/item/5ac3695656357d4ff856e16a/download

# Validation Set
echo -e "${YELLOW} Downloading Validation Data"
echo -e "${NC}"
wget --output-file=valid_data https://challenge.kitware.com/api/v1/item/5b32644c56357d41064dab4b/download

# Test Data Set
echo -e "${YELLOW} Downloading Test Data"
echo -e "${NC}"
wget --output-file=test_data https://challenge.kitware.com/api/v1/item/5b32662756357d41064dab51/download

# Unzip Training Data
echo -e "${YELLOW} Unzipping Training Data"
echo -e "${NC}"
unzip train_data
unzip train_data_truth

# Unzip Validation_Data
echo -e "${YELLOW} Unzipping Validation Data"
echo -e "${NC}"
unzip valid_data

# Unzip Test Data
echo -e "${YELLOW} Unzipping Test Data"
echo -e "${NC}"
unzip test_data

# Remove Data
rm test_data
rm train_data
rm train_data_truth
rm valid_data

echo -e "${CYAN}DONE! ALL DATA DOWNLOADED IN FILE ./MelnomaData."