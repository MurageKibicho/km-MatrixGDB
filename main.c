#include "km_ProfSafetensor.h" 
//Run: clear && gcc main.c Dependencies/cJSON.c -lm -o m.o && ./m.o
int main()
{
	char *fileName  = "Safetensors/model.safetensors"; 
	size_t fileSize = 0; 
	unsigned char *safeTensorData = kmProf_LoadSafeTensorData(fileName, &fileSize);
	assert(safeTensorData != NULL);
	
	printf("FileSize : %ld bytes\n", fileSize);
	//Get header length
	size_t headerLength = 0;
	for(int i = 7; i >= 0; i--)
	{
		headerLength <<= 8;
		headerLength += safeTensorData[i];
	}
	printf("HeaderSize : %ld bytes\n", headerLength);
	
	//Test if 8th byte is {
	assert(safeTensorData[8] == '{');
	
	//Parse tensor data with cJSON
	cJSON *tensorData = cJSON_ParseWithLength(safeTensorData+8, headerLength);
	assert(tensorData != NULL);
	
	//Load tensorData as string
	char *formatted_json = cJSON_Print(tensorData);
	assert(formatted_json != NULL);
	
	printf("%s\n",formatted_json);
	
	//Query cJSON
	size_t tensorOffsetStart = 0;
	size_t tensorOffsetEnd   = 0;
	int foundTensor = 0;
		
	foundTensor = kmProf_GetTensorOffset(tensorData, "h.4.ln_1.bias", &tensorOffsetStart, &tensorOffsetEnd);

	assert(foundTensor > -1);
	assert(tensorOffsetEnd > tensorOffsetStart);
	assert((tensorOffsetEnd - tensorOffsetStart) % 4 == 0);
	
	printf("Tensor start: %ld Tensor end: %ld\n",tensorOffsetStart, tensorOffsetEnd);
	
	//Move to weights section
	unsigned char  *weightData = (safeTensorData+8+headerLength);
	
	//Convert and print weights
	float *sampleWeights =  (float *) (weightData + tensorOffsetStart);
	for(size_t i = 0; i < (tensorOffsetEnd - tensorOffsetStart) / 4; i++)
	{
		printf("%.3f ", sampleWeights[i]);	
	}
	
	//Free formatted_json
	free(formatted_json);
	//Delete cJSON data structures
	cJSON_Delete(tensorData);
	//unmap memory
	assert(munmap(safeTensorData, fileSize) != -1);
	return 0;
}
