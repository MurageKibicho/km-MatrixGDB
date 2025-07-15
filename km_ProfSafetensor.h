#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <stdbool.h>
#include "Dependencies/cJSON.h" 

//Run: clear && gcc Safetensor.c Dependencies/cJSON.c -lm -o m.o && ./m.o
size_t kmProf_GetFileSize(char *fileName)
{
	FILE *fp = fopen(fileName, "rb");
	assert(fp != NULL);
	fseek(fp, 0L, SEEK_END);
	size_t currentFileSize = ftell(fp);rewind(fp);
	fclose(fp);
	return currentFileSize;
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

int kmProf_GetTensorOffset(cJSON *tensorData, char *tensorName, size_t *tensorStart, size_t *tensorEnd)
{
	int foundTensor = -1;
	cJSON *item = NULL;
	cJSON *offset = NULL;
	cJSON *dtype = NULL;
	cJSON *data_offsets = NULL;
	cJSON *shape = NULL;
	cJSON *eachShape = NULL;
	cJSON_ArrayForEach(item, tensorData)
	{
		dtype = cJSON_GetObjectItem(item, "dtype");data_offsets = cJSON_GetObjectItem(item, "data_offsets");
		shape = cJSON_GetObjectItem(item, "shape");
		if(dtype && data_offsets && shape)
		{
			if(strcmp(tensorName, item->string) == 0)
			{
				//printf("Key: %s\n", item->string);printf("  dtype: %s\n", dtype->valuestring);printf("  data_offsets: ");	
				cJSON_ArrayForEach(offset, data_offsets)
				{
					foundTensor += 1;
					if(foundTensor == 0)
					{
						*tensorStart = (size_t) offset->valuedouble;
					}
					else if(foundTensor == 1)
					{
						*tensorEnd = (size_t) offset->valuedouble;
					}
				}
				break;
			}
		}
	}
	return foundTensor;
}

