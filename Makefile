## ==== Dependencies
GCC=g++
NVCC=nvcc
CXXFLAGS=-std=c++14 -ltiff

## ==== Download and preprocessing
DOCKER_OUTPUT_PATH=/home/saps/output
IMAGES_DIR=./input

## Note: The following variables are used to run the docker containers and are not used in the Makefile itself
#  		 	 The container only retrieves images from between 1984 and 2017. Furthermore, not every day has  
#			 	 images of a specific region because the satellite in orbit collects images every 3 days.
# Samples:
# | landsat_8  | landsat_5  | landsat_7
# | 215065     | 215065     |
# | 2017-05-11 | 1990-03-14 | 
IMAGE_LANDSAT="landsat_8"
IMAGE_PATHROW="215065"
IMAGE_DATE="2017-05-11"

## ==== Execution
METHOD=0
THREADS=1
BLOCKS=82
OUTPUT_DATA_PATH=./output
LANDCOVER_DATA_FILE=./_empty_landcover.txt
INPUT_DATA_PATH=$(IMAGES_DIR)/$(IMAGE_LANDSAT)_$(IMAGE_PATHROW)_$(IMAGE_DATE)/final_results

clean:
	rm $(OUTPUT_DATA_PATH)/*

clean-all:
	rm -rf $(OUTPUT_DATA_PATH)/*

clean-nvidia:
	rm -rf $(OUTPUT_DATA_PATH)/nvidia/*

clean-analysis:
	rm -rf $(OUTPUT_DATA_PATH)/analysis/*

clean-images:
	rm -rf $(IMAGES_DIR)/*

build-cpp:
	$(GCC) -I./include -g ./src/cpp/*.cpp -o ./src/main $(CXXFLAGS)

build-nvcc:
	$(NVCC) -I./include -g ./src/cuda/*.cu -o ./src/main $(CXXFLAGS)

fix-permissions:
	sudo chmod -R 777 $(INPUT_DATA_PATH)/*

docker-landsat-download:
	docker run \
		-v $(IMAGES_DIR):$(DOCKER_OUTPUT_PATH) \
		-e OUTPUT_PATH=$(DOCKER_OUTPUT_PATH) \
		-e LANDSAT=$(IMAGE_LANDSAT) \
		-e PATHROW=$(IMAGE_PATHROW) \
		-e DATE=$(IMAGE_DATE) \
		cilasmarques/landsat-download:latest

docker-landsat-preprocess:
	docker run \
		-v $(IMAGES_DIR):$(DOCKER_OUTPUT_PATH) \
		-e OUTPUT_PATH=$(DOCKER_OUTPUT_PATH) \
		-e LANDSAT=$(IMAGE_LANDSAT) \
		-e PATHROW=$(IMAGE_PATHROW) \
		-e DATE=$(IMAGE_DATE) \
		cilasmarques/landsat-preprocess:latest

exec-landsat8:
	./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) & 

exec-landsat5-7:
	./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) &

nsys-landsat8:
	./bin/run-nsys.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) & 

nsys-landsat5-7:
	./bin/run-nsys.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) &

ncu-landsat5-7:
	./bin/run-ncu.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) &

ncu-landsat8:
	./bin/run-ncu.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) & 

analisys-landsat8:
	./bin/run-ana.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) &

analisys-landsat5-7:
	./bin/run-ana.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(LANDCOVER_DATA_FILE) $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) -blocks=$(BLOCKS) &