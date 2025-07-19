#define STB_DS_IMPLEMENTATION
#include "km_ProfSafetensor.h" 
//Run: clear && gcc main.c Dependencies/cJSON.c -lm -o m.o && ./m.o
int main()
{
	char *fileName  = "Safetensors/model.safetensors"; 
	size_t fileSize = 0; 
	unsigned char *safeTensorData = kmProf_LoadSafeTensorData(fileName, &fileSize);
	assert(safeTensorData != NULL);assert(fileSize > 8);assert(safeTensorData[8] == '{');
	
	size_t headerLength = kmProf_GetHeaderLength(fileSize, safeTensorData);
	//Move to weights section
	unsigned char  *weightData = (safeTensorData+8+headerLength);
	
	printf("FileSize : %ld bytes\n", fileSize);
	printf("HeaderSize : %ld bytes\n", headerLength);
	
	//Parse tensor data with cJSON
	cJSON *tensorData = cJSON_ParseWithLength(safeTensorData+8, headerLength);assert(tensorData != NULL);
	char *formatted_json = cJSON_Print(tensorData);assert(formatted_json != NULL);
	
	//Query cJSON
	printf("%s\n",formatted_json);
	KibichoTensor tensor = CreateKibichoTensor();
	SetKibichoTensor(tensorData, "h.5.attn.c_proj.weight", tensor, weightData);
	PrintKibichoTensor(tensor);
	int indices[] = {0,0};
	int indexLength = sizeof(indices) / sizeof(int);
	float item = GetTensorItem_Float(tensor, indexLength, indices);
	
	DestroyKibichoTensor(tensor);
	//Free formatted_json
	free(formatted_json);
	//Delete cJSON data structures
	cJSON_Delete(tensorData);
	//unmap memory
	assert(munmap(safeTensorData, fileSize) != -1);
	return 0;
}
