#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <stdbool.h>
#include "Dependencies/cJSON.h" 
#include "Dependencies/stb_ds.h" 
//Run: clear && gcc Safetensor.c Dependencies/cJSON.c -lm -o m.o && ./m.o
typedef struct kibicho_tensor_struct *KibichoTensor;
struct kibicho_tensor_struct
{
	int size;
	int dimensionCount;
	int foundKibichoTensor;
	size_t offsetStart;
	size_t offsetEnd;
	int *shape;
	int *strides;
	float *data;
};

KibichoTensor CreateKibichoTensor()
{
	KibichoTensor tensor = malloc(sizeof(struct kibicho_tensor_struct));
	tensor->dimensionCount = 0;
	tensor->size = 0;
	tensor->dimensionCount = 0;
	tensor->foundKibichoTensor = -1;
	tensor->offsetStart = 0;
	tensor->offsetEnd = 0;
	tensor->shape = NULL;
	tensor->strides = NULL;
	tensor->data = NULL;
	return tensor;
}

void SetKibichoTensor(cJSON *tensorData, char *tensorName, KibichoTensor tensor, unsigned char *weightData)
{
	//Reset KibichoTensor
	tensor->foundKibichoTensor = -1;tensor->dimensionCount = 0;arrsetlen(tensor->shape, 0);arrsetlen(tensor->strides, 0);
	cJSON *item = NULL;cJSON *offset = NULL;cJSON *dtype = NULL;cJSON *data_offsets = NULL;cJSON *shape = NULL;cJSON *eachShape = NULL;
	cJSON_ArrayForEach(item, tensorData)
	{
		dtype = cJSON_GetObjectItem(item, "dtype");
		data_offsets = cJSON_GetObjectItem(item, "data_offsets");
		shape = cJSON_GetObjectItem(item, "shape");
		if(dtype && data_offsets && shape)
		{
			if(strcmp(tensorName, item->string) == 0)
			{
				//printf("Key: %s\n", item->string);printf("  dtype: %s\n", dtype->valuestring);printf("  data_offsets: ");	
				cJSON_ArrayForEach(eachShape, shape)
				{
					arrput(tensor->shape, eachShape->valueint);
					tensor->dimensionCount += 1;
				}
				cJSON_ArrayForEach(offset, data_offsets)
				{
					tensor->foundKibichoTensor += 1;
					if(tensor->foundKibichoTensor == 0)
					{
						tensor->offsetStart = (size_t) offset->valuedouble;
					}
					else if(tensor->foundKibichoTensor == 1)
					{
						tensor->offsetEnd   = (size_t) offset->valuedouble;
					}
				}
				break;
			}
		}
	}
	
	//Set strides
	int stride = 1;
	tensor->size = 1;
	arrsetlen(tensor->strides, tensor->dimensionCount);
	for(int i = 0, j = tensor->dimensionCount - 1; i < tensor->dimensionCount && j > -1; i++, j--)
	{
		tensor->size *= tensor->shape[i];
		tensor->strides[j] = stride;
		stride *= tensor->shape[j];
	}
	
	//Set Weights
	tensor->data =  (float *) (weightData + tensor->offsetStart);
}

void PrintKibichoTensor(KibichoTensor tensor)
{
	if(tensor && tensor->foundKibichoTensor > -1)
	{
		printf("Size: %d, Dimensions: %d\nOffsets[%ld,%ld]\nShape[%d", tensor->size,tensor->dimensionCount, tensor->offsetStart,tensor->offsetEnd,tensor->shape[0]);
		for(int i = 1; i < tensor->dimensionCount; i++)
		{
			printf(",%d", tensor->shape[i]);
		}
		printf("]\nStrides[%d",tensor->strides[0]);
		for(int i = 1; i < tensor->dimensionCount; i++)
		{
			printf(",%d", tensor->strides[i]);
		}
		printf("]\n");
	}
}
void DestroyKibichoTensor(KibichoTensor tensor)
{
	if(tensor && tensor->foundKibichoTensor > -1)
	{
		if(tensor->shape){arrfree(tensor->shape);}
		if(tensor->strides){arrfree(tensor->strides);}
		free(tensor);
	}
}

size_t kmProf_GetFileSize(char *fileName)
{
	FILE *fp = fopen(fileName, "rb");
	assert(fp != NULL);
	fseek(fp, 0L, SEEK_END);
	size_t currentFileSize = ftell(fp);rewind(fp);
	fclose(fp);
	return currentFileSize;
}

void kmProf_PrintIntArray(int length, int *array)
{
	for(int i = 0; i < length; i++)
	{
		printf("%3d, ", array[i]);
	}
	printf("\n");
}

unsigned char *kmProf_LoadSafeTensorData(char *fileName, size_t *fileSizeHolder)
{
	size_t fileSize = kmProf_GetFileSize(fileName);
	FILE *fp = fopen(fileName, "rb");assert(fp != NULL);
	int fileNumber = fileno(fp);
	unsigned char *fileData = mmap(NULL,fileSize, PROT_READ, MAP_PRIVATE, fileNumber, 0);assert(fileData != NULL);
	assert(fileData != MAP_FAILED);
	fclose(fp);
	*fileSizeHolder = fileSize;
	return fileData;
}

size_t kmProf_GetHeaderLength(size_t fileSize, unsigned char *safeTensorData)
{
	assert(fileSize > 8);
	size_t headerLength = 0;
	for(int i = 7; i >= 0; i--)
	{
		headerLength <<= 8;
		headerLength += safeTensorData[i];
	}
	return headerLength;
}

float GetTensorItem_Float(KibichoTensor tensor, int indexLength, int *indices)
{
	if(tensor)
	{
		assert(indexLength == tensor->dimensionCount);
		assert(tensor->strides);assert(tensor->data);
		int index = 0;
		for(int i = 0; i < tensor->dimensionCount; i++)
		{
			assert(indices[i] >= 0);
			assert(indices[i] < tensor->shape[i]);
			index += indices[i] * tensor->strides[i];
		}
		assert(index > -1);
		assert(index < tensor->size);
		return tensor->data[index];
	}
}

