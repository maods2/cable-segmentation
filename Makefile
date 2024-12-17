# Variables
CONFIG_FILE=config.yaml

download-dataset:
	gdown https://drive.google.com/uc?id=128Bjry_c_8BOEf2FgsXDQqz7pYAIk3Da 
	unzip data_original_size.zip
	rm -rf data_original_size.zip

download-dataset-cleaned:
	gdown https://drive.google.com/uc?id=1WNx0-Yv7vLkhoYI4NqO2hXjIo5NwSPgW 
	unzip data_wire_clean.zip
	rm -rf data_wire_clean.zip

# Main target: runs training
train:
	@echo "Running Python script with configuration file $(CONFIG_FILE)"
	python3 src/main.py --train --config "./configs/$(CONFIG_FILE)"

train-sam:
	python3 src/sam_model.py --num_epochs 1 --batch_size 2 

inference:
	python3 src/inference.py 


	https://drive.google.com/file/d/1WNx0-Yv7vLkhoYI4NqO2hXjIo5NwSPgW/view?usp=sharing
	