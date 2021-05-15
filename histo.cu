
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include <math.h>



#define DEFAULT_FILENAME "small-zibra-unsplash.ppm"

#define MAX_VALUE 256  //max value of the pic luminance
#define NUM_BINS 256 //num of bins equals to the max value
__constant__ double PARAMS[4];

void write_ppm( char const *filename, int xsize, int ysize, int maxval, unsigned int *pic);
unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval );
void write_CSV(char const *filename,int width, int height, unsigned int *input);
void matrixRotation(unsigned int *input, unsigned int *output, int width, int height, double angle);
void computerGoldHisto(unsigned int* input, unsigned int* histo, int width, int height);
void getNewXY(int inputX, int inputY, int width, int height,double angle, int *outputX, int *outputY);

__global__ void rotation_kernel_naive(unsigned int *input,unsigned int *output,int width, int height,double angle);
__global__ void rotation_kernel_2(unsigned int *input,unsigned int *output,int width, int height,double angle);
__global__ void rotation_kernel_3(unsigned int *input,unsigned int *output,int width,int height);

__global__ void histo_kernel_naive(unsigned int *input,unsigned int *histo,int width, int height);
__global__ void histo_kernel_2(unsigned int *input,unsigned int *histo,int width, int height);


int main( int argc, char **argv )
{
	double ang ;
	char *filename;
	filename = strdup( DEFAULT_FILENAME);
	ang = 45.0;
	if (argc > 1) {
		if (argc == 3)  { //angle and filename

			ang = atoi( argv[1] );
			filename = strdup( argv[2]);
		}
		if (argc == 2) { // angle
			ang = atoi( argv[1] );
		}

	}
		fprintf(stderr, "file %s , rotation angle: %f\n", filename, ang);
  //initialization paramters
  int xsize, ysize,maxval;

  unsigned int *h_histoCPU;
  unsigned int *h_Input;
  unsigned int *h_rotated;

  //read input from image
  unsigned int *pic = read_ppm( filename, &xsize, &ysize, &maxval );


  int diaLen = (unsigned int) (sqrt(xsize * xsize + ysize* ysize) + 3);//paddle extra 3 for safer non-cropped rotation
  printf("width:%d, height:%d,maxVal: %d, diagonal size: %d \n",xsize,ysize,maxval,diaLen );
  //decide memory size
  size_t histo_size = MAX_VALUE * diaLen * 3 * sizeof(int);
  size_t rotate_size = diaLen * diaLen * 3 * sizeof(int);

  //allocate memory

  h_histoCPU = (unsigned int*)malloc(histo_size);
  h_rotated= (unsigned int*)malloc(rotate_size);
  h_Input = (unsigned int*)malloc(rotate_size);

  if (!h_Input || !h_histoCPU || !h_rotated) {
		fprintf(stderr, " unable to malloc \n");
		exit(-1); // fail
	}

  //decide rotate angle， rotation is done swirlly along the image center.
 double angle = - ang / 360 * M_PI * 2;
//paddle data for h_Input, make it squre with side length as the input diagonal length
    int deltaX = diaLen - xsize ;
    int deltaY = diaLen - ysize ;

    for(int i=0;i<diaLen;i++){
      for(int j = 0; j< diaLen;j++){
        if(i>=deltaY / 2 && i< (ysize + deltaY/2) && j >= deltaX/2 && j < (xsize + deltaX/2)){
          h_Input[i*diaLen+j]=pic[(i-deltaY/2)*xsize + (j-deltaX/2)];
        }else{
          h_Input[i*diaLen+j] = 0;
        }
      }
    }
//calculate new coordinate

int outputX, outputY;
double realAngle = -angle;
int originX = 0;
int originY = 0;
getNewXY(originX,originY, xsize, ysize, realAngle, &outputX, &outputY);
printf("(%d, %d) rotated to newX: %d, newY: %d \n", originX,originY,outputX, outputY);
//output the paddledInput
write_ppm( "paddledInput.ppm", diaLen, diaLen, 255, h_Input);
//timer for cpu rotation------------------------
cudaEvent_t start, stop;
float time;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
//run rotation on cpu
matrixRotation( h_Input, h_rotated, diaLen, diaLen, angle);

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time,start,stop);

printf("cpu rotation time: %f milliseconds\n",time);
cudaEventDestroy(start);
cudaEventDestroy(stop);
//output rotated image
write_ppm( "rotated_gold.ppm", diaLen, diaLen, 255, h_rotated);
//write_CSV("rotated_gold.csv",diaLen,diaLen, h_rotated);
//write_CSV("picture.csv",xsize,ysize, pic);

//initialization h_BinsCPU
for(int i = 0;i<NUM_BINS;i++){
  for(int j = 0;j<MAX_VALUE;j++){
    h_histoCPU[i*MAX_VALUE+j] = 0;
  }
}
//timer for CPU histogram
float time2;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);

//run histo on CPU
computerGoldHisto(h_rotated,h_histoCPU,diaLen,diaLen);

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time2,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
printf("cpu histo time: %f milliseconds\n",time2);

//output histo result to CSV file
write_CSV("histoBinCPU.csv",diaLen,MAX_VALUE, h_histoCPU);

//cudaMalloc for rotation kernel
unsigned int *h_histoGPU;
unsigned int *h_rotatedGPU;
unsigned int *h_rotatedGPU2;
unsigned int *h_rotatedGPU3;
unsigned int *d_Input;
unsigned int *d_rotated;
unsigned int *d_rotated_naive;
unsigned int *d_histo;


//memory allocate in the host
h_rotatedGPU = (unsigned int*)malloc(rotate_size);
h_rotatedGPU2 = (unsigned int*)malloc(rotate_size);
h_rotatedGPU3 = (unsigned int*)malloc(rotate_size);
h_histoGPU = (unsigned int*)malloc(histo_size);
if (!h_rotatedGPU ||!h_histoGPU||!h_rotatedGPU2||!h_rotatedGPU3) {
  fprintf(stderr, " unable to malloc \n");
  exit(-1); // fail
}

cudaMalloc((void**)&d_Input,rotate_size);
cudaMalloc((void**)&d_rotated,rotate_size);
cudaMalloc((void**)&d_rotated_naive,rotate_size);

//cudaMemcpy
cudaMemcpy(d_Input,h_Input,rotate_size,cudaMemcpyHostToDevice);

//kernel dimension
int blockSize = 32;
dim3 blockDim(blockSize,blockSize,1);
int gridSize = (diaLen + blockSize -1)/blockSize;
dim3 gridDim(gridSize,gridSize,1);
float time3;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);

//launch kernel 100 times and take average time
for(int i = 0;i<100;i++){
	rotation_kernel_naive<<<gridDim,blockDim>>>(d_Input,d_rotated_naive,diaLen,diaLen,angle);
}
cudaEventRecord(stop,0);
cudaMemcpy(h_rotatedGPU,d_rotated_naive,rotate_size,cudaMemcpyDeviceToHost);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time3,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
printf("GPU rotation_kernel_naive time: %f milliseconds\n",time3 / 100);
//output GPU rotate result
write_ppm( "rotated_GPU_naive.ppm", diaLen, diaLen, 255, h_rotatedGPU);
//GPU rotation with optimization, using registers to store pre-calculated values

cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
//launch kernel 100 times and take average time
for(int i = 0;i<100;i++){
rotation_kernel_2<<<gridDim,blockDim>>>(d_Input,d_rotated,diaLen,diaLen,angle);
}
cudaEventRecord(stop,0);
cudaMemcpy(h_rotatedGPU2,d_rotated,rotate_size,cudaMemcpyDeviceToHost);
cudaEventSynchronize(stop);
float time3_2;
cudaEventElapsedTime(&time3_2,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
printf("GPU rotation_kernel_2 time: %f milliseconds\n",time3_2 / 100);

//output GPU rotate result
write_ppm( "rotated_GPU2.ppm", diaLen, diaLen, 255, h_rotatedGPU2);
//rotation kernel_3 using constant memory---------------------------------
double *P;
P = (double*) malloc(4 * sizeof(double));
P[0] = (double)diaLen / 2; //xCenter
P[1] = (double)diaLen / 2; //yCenter
P[2] = sin(angle);
P[3] = cos(angle);

//load data to constant memory
cudaMemcpyToSymbol(PARAMS, P, 4 * sizeof(double));
//timer
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
//launch kernel 100 times and take average time
for(int i = 0;i<100;i++){
rotation_kernel_3<<<gridDim,blockDim>>>(d_Input,d_rotated,diaLen,diaLen);
}
cudaEventRecord(stop,0);
cudaMemcpy(h_rotatedGPU3,d_rotated,rotate_size,cudaMemcpyDeviceToHost);
cudaEventSynchronize(stop);
float time3_3;
cudaEventElapsedTime(&time3_3,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
printf("GPU rotation kernel_3 tiem: %f milliseconds\n",time3_3 / 100);

//output GPU rotate result
write_ppm( "rotated_GPU3.ppm", diaLen, diaLen, 255, h_rotatedGPU3);
//-------end of rotation kernel-----------------------------------------------
//cudaFree part 1
cudaFree(d_Input);
cudaFree(d_rotated_naive);

//------GPU histo start----------------------------------------

// //cudaMalloc for histoGPU
cudaMalloc((void**)&d_histo,histo_size);
//------naive histo kernel-----------------------------------------
float time4;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);

//launch kernel 100 times and take average time
int blockSize2 = 32;
int gridSize2 = ceil((float)diaLen/blockSize2);
dim3 blockDim2 (blockSize2,blockSize2,1);
dim3 gridDim2 (gridSize2,gridSize2,1);

histo_kernel_naive<<<gridDim2,blockDim2>>>(d_rotated,d_histo,diaLen,diaLen);

cudaEventRecord(stop,0);
cudaMemcpy(h_histoGPU,d_histo,histo_size,cudaMemcpyDeviceToHost);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time4,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
printf("GPU histo_naive time: %f milliseconds\n",time4 );

// //output histo result to CSV file
 write_CSV("histoBinGPU_kernel_naive.csv",diaLen,MAX_VALUE, h_histoGPU);
//----------end---------------------------------------------------------------

//------histo kernel 2, one thread work with one input----------------------------
unsigned int *h_histoGPU2;
h_histoGPU2 = (unsigned int*)malloc(histo_size);
if (!h_histoGPU2) {
  fprintf(stderr, " unable to malloc \n");
  exit(-1); // fail
}
unsigned int *d_histo2;
cudaMalloc((void**)&d_histo2,histo_size);

float time5;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
//launch kernel 100 times and take average time
for(int i = 0;i<100;i++){
histo_kernel_2<<<diaLen,256>>>(d_rotated,d_histo2,diaLen,diaLen);
}
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaMemcpy(h_histoGPU2,d_histo2,histo_size,cudaMemcpyDeviceToHost);

cudaEventElapsedTime(&time5,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
printf("GPU histo_kernel_2 time: %f milliseconds\n",time5 / 100);

// //output histo result to CSV file
 write_CSV("histoBinGPU_kernel_2.csv",diaLen,MAX_VALUE, h_histoGPU2);
//----------end---------------------------------------------------------------



//cudaFREE part 2
cudaFree(d_rotated);
cudaFree(d_histo);
cudaFree(d_histo2);


//Free host memory
free(h_Input);
free(h_rotated);
free(h_rotatedGPU);
free(h_rotatedGPU2);
free(h_rotatedGPU3);
free(h_histoCPU);
free(h_histoGPU);
free(h_histoGPU2);



fprintf(stderr, "done\n");

}
//--------rotation kernel naive----------------------------------------
__global__ void rotation_kernel_naive(unsigned int *input,unsigned int *output,int width, int height,double angle){
  //TO DO
  double xCenter = (double)width / 2;
  double yCenter = (double)height / 2;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * width + x;
  int orgX = 0;
  int orgY = 0;
  //boundary check
  if(x >=0 && x < width && y >=0 && y < height){
    orgX = (int)(cos(angle) * ((double)x - xCenter)- sin(angle) * ((double)y - yCenter) + xCenter);
    orgY = (int)(sin(angle) * ((double)x - xCenter) + cos(angle) * ((double)y - yCenter) + yCenter);
  }

  if(orgX>=0 && orgX < width && orgY>=0 && orgY<height){
    output[index] = input[ orgY * width + orgX];
  }

}
//--------rotation kernel 2----------------------------------------
__global__ void rotation_kernel_2(unsigned int *input,unsigned int *output,int width, int height,double angle){
  //TO DO
  double xCenter = (double)width / 2;  // x center of image
  double yCenter = (double)height / 2; // y center of image
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * width + x;
  double sinA = sin(angle);
  double cosA = cos(angle);
  double shiftX = (double)x - xCenter;
  double shiftY = (double)y - yCenter;
  int orgX = 0;
  int orgY = 0;
  //boundary check
  if(x >=0 && x < width && y >=0 && y < height){
    orgX = (int)(cosA * shiftX - sinA * shiftY + xCenter);
    orgY = (int)(sinA * shiftX + cosA * shiftY + yCenter);
  }

  if(orgX>=0 && orgX < width && orgY>=0 && orgY<height){
    output[index] = input[ orgY * width + orgX];
  }

}

//--------rotation kernel 3 using constant memory----------------------------------------
__global__ void rotation_kernel_3(unsigned int *input,unsigned int *output,int width, int height){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * width + x;

  double shiftX = (double)x - PARAMS[0];
  double shiftY = (double)y - PARAMS[1];
  int orgX = 0;
  int orgY = 0;
  //boundary check
  if(x >=0 && x < width && y >=0 && y < height){
    orgX = (int)(PARAMS[3] * shiftX - PARAMS[2] * shiftY + PARAMS[0]);
    orgY = (int)(PARAMS[2] * shiftX + PARAMS[3] * shiftY + PARAMS[1]);
  }

  if(orgX>=0 && orgX < width && orgY>=0 && orgY<height){
    output[index] = input[ orgY * width + orgX];
  }

}

//----------cpu for matrix rotation--------------------------
//rotate as a swirl from , from the center
void matrixRotation(unsigned int *input, unsigned int *output, int width, int height, double angle){
   double xCenter = (double)width / 2;
   double yCenter = (double)height / 2;
  //for non-crop rotation put both the height and width of the output the diagonal length of the origin input
  for(int y = 0;y<height;y++){
    for(int x = 0;x<width;x++){
      int orgX = (int)(cos(angle) * ((double)x -xCenter) - sin(angle) * ((double)y - yCenter) + xCenter ) ;
      int orgY = (int)(sin(angle) * ((double)x -xCenter) + cos(angle) * ((double)y - yCenter) + yCenter );
      if(orgX>=0 && orgX < width && orgY>=0 && orgY < height){
        output[y*width+x] = input[orgY * width +orgX];

      }
    }
  }

}


//----------histo kernel naive: using global memory--------------------------------------
__global__ void histo_kernel_naive(unsigned int *input,unsigned int *histo,int width, int height){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

if(x >=0 && x < width && y >=0 && y < height){
		atomicAdd(&histo[input[y*width+x]*width+x],1);
	}

}
//histo_kernel_2<<<diaLen,256>>>(d_rotated,d_histo2,diaLen,diaLen);
//----------histo kernel 2: with shared memory,each threads deal with one input
__global__ void histo_kernel_2(unsigned int *input,unsigned int *histo,int width, int height){
 __shared__ unsigned int H[NUM_BINS];// one column of the output bin
 int tx = threadIdx.x;
 int col = blockIdx.x;//col number of input
 int bd = blockDim.x;
	H[tx] = 0;
 __syncthreads();
  for(int t =0;t<(height+bd-1)/bd;t++){ //when height is bigger than blockDim
		 int row = t*bd+tx;
		 if(row<height){
			 int value = input[row*width+col];
				atomicAdd(&H[value],1);
		 }
	}
	__syncthreads();
		histo[tx*width+col] = H[tx];
}
//histo_kernel_3<<<diaLen,256>>>,
//----------histo kernel 3: with shared memory for input, each thread collecting one bin
__global__ void histo_kernel_3(unsigned int *input,unsigned int *histo,int width, int height){

 __shared__ unsigned int IN[NUM_BINS]; //use to storing input
 int tx = threadIdx.x;
 int col = blockIdx.x;//col number of input

	int h = 0;
  for(int t =0;t<(height+NUM_BINS-1)/NUM_BINS;t++){ //when height is bigger than NUM_BINS
		 int row = t*NUM_BINS+tx;
		 if(row<height){
			 IN[tx] = input[row*width+col];
		 }else{
			 IN[tx] = 300;  // padding nonsense value to avoid addition
		 }
			 __syncthreads();
			 for(int i = 0;i<NUM_BINS;i++){//each thread loop through the input
				 int value = IN[i];
				 if(tx == value){
					 h++;
				 }
			 }
	}

   histo[tx*width+col] = h;
}



//----------cpu for histo along y direction-------------------
void computerGoldHisto(unsigned int* input, unsigned int* histo, int width, int height){

  for(int i = 0; i<height;i++){
    for(int j = 0;j<width;j++){
        int data = input[i*width+j];
        histo[data*width+j]++;
    }
  }
}


//-----------read image to array-----------------------------------------------------------------
unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){

	if ( !filename || filename[0] == '\0') {
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

	FILE *fp;

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail
	}

	char chars[1024];
	//int num = read(fd, chars, 1000);
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6')
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line!
	{
		ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;

	unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}

	// really read
	char duh[80];
	char *line = chars;

	// find the start of the pixel data.
	sprintf(duh, "%d\0", *xsize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *ysize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *maxval);
	line = strstr(line, duh);

	fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
	line += strlen(duh) + 1;

	long offset = line - chars;
	//lseek(fd, offset, SEEK_SET); // move to the correct offset
	fseek(fp, offset, SEEK_SET); // move to the correct offset
	//long numread = read(fd, buf, bufsize);
	long numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize);

	fclose(fp);

	int pixels = (*xsize) * (*ysize);
	for (int i=0; i<pixels; i++)
		pic[i] = (int) buf[3*i];  // red channel

	return pic; // success
}



//--------------wiret array to a image-------------------------------------------------------------------
void write_ppm( char const *filename, int xsize, int ysize, int maxval,unsigned int *pic)
{
	FILE *fp;
// 	int x,y;

	fp = fopen(filename, "wb");
	if (!fp)
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
		exit(-1);
	}

	fprintf(fp, "P6\n");
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

	int numpix = xsize * ysize;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc);
	}

	fclose(fp);
}

//write histoBin result to excel  diaLen, MAX_VALUE
void write_CSV(char const *filename,int width, int height, unsigned int *input){
  FILE *fp;
  fp = fopen(filename, "w+");
  if (!fp)
  {
    fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
    exit(-1);
  }
 for(int i = 0;i< height;i++){
   for(int j = 0;j<width;j++){
     fprintf(fp,"%d,",input[i*width+j]);
   }
   fprintf(fp,"\n");
 }
 fclose(fp);

}
// calculate newX newY after padding and rotation
void getNewXY(int inputX, int inputY, int width, int height,double angle, int *outputX, int *outputY){
  double diaLen = (int)(sqrt(width * width + height * height) + 3);
	double deltaX = diaLen - width ;
	double deltaY = diaLen - height;
//  printf("deltaX: %f, deltaY: %f\n",deltaX,deltaY);
  double x = (double)inputX - deltaX / 2;
  double y = (double)inputY - deltaY/ 2;
//  printf("paddled x and y are: %f, %f\n", x, y);
  double xCenter = -diaLen / 2;
  double yCenter = -diaLen / 2;
//	printf("deltaX: %f, deltaY: %f\n",xCenter,yCenter);

  *outputX = -1 * (int)(cos(angle) * (x -xCenter) - sin(angle) * (y - yCenter) + xCenter ) ;
  *outputY = -1 * (int)(sin(angle) * (x -xCenter) + cos(angle) * (y - yCenter) + yCenter );


}
