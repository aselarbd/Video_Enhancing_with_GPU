/*

you can compile and run the code in VStudio configuring the OPENCV
make sure opencv is in you computer

or    
you can compile the code as  "nvcc  SoftwareGPU.cu helpers.cu `pkg-config --cflags --libs opencv`" 

run it   ./a.out   in  Linux



*/

#include <stdint.h>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <time.h>
#include "helpers.cuh"
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

uchar* grayDataPtr;
uchar* hsvDataPtr;
uchar *arr;
int dWidth, dHeight;


__global__ void histogram_hsvtorgb(uchar *f_out1,uchar *f_in1,uchar *f_ini,int rows,int cols){
    float fR, fG, fB;
    float fH, fS, fV;
    const float FLOAT_TO_BYTE = 255.0f;
    const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
    int i,j;

    //derive the row and column based on thread configuration
    i = blockIdx.y*blockDim.y + threadIdx.y;
    j = blockIdx.x*blockDim.x + threadIdx.x;
   
    //Limit calculations for valid indices
    if(i < rows && j < cols){
       
		// Get the HSV pixel components
		uchar bH = f_in1[3 * (cols*i + j)];// H component
		uchar bS = f_in1[3 * (cols*i + j) + 1];    // S component
		uchar bV = f_ini[(cols*i + j)];    // V component

		// Convert from 8-bit integers to floats
		fH = (float)bH * BYTE_TO_FLOAT;
		fS = (float)bS * BYTE_TO_FLOAT;
		fV = (float)bV * BYTE_TO_FLOAT;

		// Convert from HSV to RGB, using float ranges 0.0 to 1.0
		int iI;
		float fI, fF, p, q, t;

        if (bS == 0) {
            // achromatic (grey)
            fR = fG = fB = fV;
        }
        else {
			// If Hue == 1.0, then wrap it around the circle to 0.0
            if (fH >= 1.0f)	fH = 0.0f;

            fH *= 6.0;            // sector 0 to 5
			fI = floor(fH);        // integer part of h (0,1,2,3,4,5 or 6)
			iI = (int)fH;            //        "        "        "        "
			fF = fH - fI;            // factorial part of h (0 to 1)

			p = fV * (1.0f - fS);
			q = fV * (1.0f - fS * fF);
			t = fV * (1.0f - fS * (1.0f - fF));

			switch (iI) {
				case 0:
				fR = fV;
				fG = t;
				fB = p;
				break;
				case 1:
				fR = q;
				fG = fV;
				fB = p;
				break;
				case 2:
				fR = p;
				fG = fV;
				fB = t;
				break;
				case 3:
				fR = p;
				fG = q;
				fB = fV;
				break;
				case 4:
				fR = t;
				fG = p;
				fB = fV;
				break;
				default:       
				fR = fV;
				fG = p;
				fB = q;
				break;
			}
		}

		// Convert from floats to 8-bit integers
		uchar bR = (uchar)(fR * FLOAT_TO_BYTE);
		uchar bG = (uchar)(fG * FLOAT_TO_BYTE);
		uchar bB = (uchar)(fB * FLOAT_TO_BYTE);

		// Clip the values to make sure it fits within the 8bits.
		if (bR > 255)	bR = 255;
		if (bG > 255)	bG = 255;
		if (bB > 255)	bB = 255;
		
		// Set the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
		f_out1[3 * (cols*i + j)] = bB;        // B component
		f_out1[3 * (cols*i + j)+1] = bG;        // G component
		f_out1[3 * (cols*i + j)+2] = bR;        // R component

    }

}

__global__ void histogram_rgbtohsv(uchar *f_out,uchar *f_outi,uchar *f_in,int rows,int cols){
   
    int i,j;
    float fR, fG, fB;
    float fH, fS, fV;
    const float FLOAT_TO_BYTE = 255.0f;
    const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;

    //derive the row and column based on thread configuration
    i = blockIdx.y*blockDim.y + threadIdx.y;
    j = blockIdx.x*blockDim.x + threadIdx.x;
   
    //Limit calculations for valid indices
    if(i < rows && j < cols){
       
       
        // Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
        uchar bB =f_in[3 * (cols*i + j)];    // Blue component
        uchar bG = f_in[3 * (cols*i + j) + 1];    // Green component
        uchar bR = f_in[3 * (cols*i + j) + 2];    // Red component
		uchar intensity = (bG + bR + bB) / 3;
				
		// Convert from 8-bit integers to floats.
		fR = bR * BYTE_TO_FLOAT;
		fG = bG * BYTE_TO_FLOAT;
		fB = bB * BYTE_TO_FLOAT;

		// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
		float fDelta;
		float fMin, fMax;
		uchar iMax;
		// Get the min and max, but use integer comparisons for slight speedup.
		if (bB < bG) {
			if (bB < bR) {
				fMin = fB;
				if (bR > bG) {
					iMax = bR;
					fMax = fR;
				}
				else {
					iMax = bG;
					fMax = fG;
				}
			}
			else {
				fMin = fR;
				fMax = fG;
				iMax = bG;
			}
		}
		else {
			if (bG < bR) {
				fMin = fG;
				if (bB > bR) {
					fMax = fB;
					iMax = bB;
				}
				else {
					fMax = fR;
					iMax = bR;
				}
			}
			else {
				fMin = fR;
				fMax = fB;
				iMax = bB;
			}
		}
		fDelta = fMax - fMin;
		fV = fMax;                // Value (Brightness).
		if (iMax != 0) {            // Make sure it's not pure black.
			fS = fDelta / fMax+0.00001;        // Saturation.
			float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta + 0.00001);    // Make the Hues between 0.0 to 1.0 instead of 6.0
			if (iMax == bR) {        // between yellow and magenta.
				fH = (fG - fB) * ANGLE_TO_UNIT;
			}
			else if (iMax == bG) {        // between cyan and yellow.
				fH = (2.0f / 6.0f) + (fB - fR) * ANGLE_TO_UNIT;
			}
			else {                // between magenta and cyan.
				fH = (4.0f / 6.0f) + (fR - fG) * ANGLE_TO_UNIT;
			}
                    // Wrap outlier Hues around the circle.
			if (fH < 0.0f)
				fH += 1.0f;
			if (fH >= 1.0f)
				fH -= 1.0f;
		}
		else {
			// color is pure Black.
			fS = 0;
			fH = 0;    // undefined hue
		}

		// Convert from floats to 8-bit integers.
		uchar bH = (uchar)(0.5f + fH * 255.0f);
		uchar bS = (uchar)(0.5f + fS * 255.0f);
		uchar bV = (uchar)(0.5f + fV * 255.0f);

		// Clip the values to make sure it fits within the 8bits.
		if (bH > 255)
			bH = 255;
		if (bS > 255)
			bS = 255;
		if (bV > 255)
			bV = 255;
		
		// Set the HSV pixel components.
		f_out[3 * (cols*i + j)] = bH;        // H component
		f_out[3 * (cols*i + j) + 1] = bS;        // S component
		f_outi[(cols*i + j) ] = intensity;
   
    }
   
   
}


__global__ void histogram_equlization(uchar *out,uchar *in,int * in2,int rows,int cols){
   
    int i,j;
   
    //derive the row and column based on thread configuration
    i = blockIdx.y*blockDim.y + threadIdx.y;
    j = blockIdx.x*blockDim.x + threadIdx.x;
   
    //Limit calculations for valid indices
    if(i < rows && j < cols){
        const int gi = i*cols + j;
        out[gi] = in2[(int)in[gi]];
    }
}

uchar * histwqualize(unsigned char * image, int image_rows, int image_cols)
{
    int i, y, x;
    uchar  *arr1=image;
	
	
    // Generate the histogram
    int histogram[256];
    for (i = 0; i < 256; i++)
    {
        histogram[i] = 0;
    }


    // calculate the no of pixels for each intensity values
    for (y = 0; y < image_rows; y=y+10)
        for (x = 0; x < image_cols; x=x+10)
            histogram[(int)image[y*image_cols + x]]++;

	// Caluculate the size of image
    int size = image_rows * image_cols/100;
    float alpha =(float) 255.0 / size;

    // Calculate the probability of each intensity
    float PrRk[256];
    for (i = 0; i < 256; i++)
    {
        PrRk[i] = (float)histogram[i] / size;
    }

    // Generate cumulative frequency histogram
    int cumhistogram[256];
    cumhistogram[0] = histogram[0];

    for (i = 1; i < 256; i++)
    {
        cumhistogram[i] = histogram[i] + cumhistogram[i - 1];
    }


    // Scale the histogram
    int Sk[256];
    for (i = 0; i < 256; i++)
    {
        Sk[i] = (int)(cumhistogram[i] * alpha);
    }

    // Generate the equlized histogram
    float PsSk[256];
    for (i = 0; i < 256; i++)
    {
        PsSk[i] = 0;
    }

    for (i = 0; i < 256; i++)
    {
        PsSk[Sk[i]] += PrRk[i];
    }

    // Generate the equlized image

        uchar *c_in;
        uchar *c_out;
        int * c_in2;
        cudaMalloc((void **)&c_in,  sizeof(uchar)*dHeight*dWidth ); 
        cudaMalloc((void **)&c_out, sizeof(uchar)*dHeight*dWidth ); 
        cudaMalloc((void **)&c_in2, sizeof(int)*256 ); 
       
        //copy memory from ram to cuda
        cudaMemcpy(c_in, image , sizeof(uchar)*dHeight*dWidth, cudaMemcpyHostToDevice ); 
        cudaMemcpy(c_in2, Sk , sizeof(int)*256, cudaMemcpyHostToDevice ); 
       
        dim3 threadsPerBlock(16,16);
        dim3 numBlocks(ceil(dWidth/(float)16),ceil(dHeight/(float)16));
        histogram_equlization<<<numBlocks,threadsPerBlock>>>(c_out,c_in,c_in2,dHeight,dWidth);
        /*kernel calls are asynchronous. Hence the checkCudaError() function will execute before the kernel finished.
        In order to tell wait till the kernel is over we use cudaDeviceSynchronize() before checkCudaError()
        In previous error checking examples it should be corrected as this*/
        cudaDeviceSynchronize(); 
       
       
       
        //copy the answer back from cuda to ram
        cudaMemcpy(arr1, c_out, sizeof(uchar)*dHeight*dWidth, cudaMemcpyDeviceToHost ); 

        //free the cuda memory
        cudaFree(c_in); 
        cudaFree(c_out); 
        cudaFree(c_in2); 
   
    
    return arr1;
}





int main(int, char**)
{
	
		cudaEvent_t start,stop;
		float elapsedtime;
		cudaEventCreate(&start);
		cudaEventRecord(start,0);
	
    VideoCapture cap("lowcon.mp4"); // here you can Input the video path
       
   
    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video file" << endl;
        return -1;
    }
    dWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    dHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

   
    hsvDataPtr = new uchar[dWidth*dHeight * 3];
	grayDataPtr = new uchar[dWidth*dHeight];

    //this is for saving the outputvideo
    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
    VideoWriter oVideoWriter("MyVideo.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, frameSize, true); //initialize the VideoWriter object

    if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }
   
      
   while (1)
    {
		
        Mat frame, frame1;
        Mat grayImg, hsvImg;

        bool bSuccess = cap.read(frame); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }
 
      //pointers for memory allocation in cudaa
        uchar *f_in;
        uchar *f_out;
		uchar *f_outi;
       
        //allocate memory in cuda
        cudaMalloc((void **)&f_in,  sizeof(uchar)*dHeight*dWidth*3 ); 
        cudaMalloc((void **)&f_out, sizeof(uchar)*dHeight*dWidth*3 ); 
        cudaMalloc((void **)&f_outi, sizeof(uchar)*dHeight*dWidth); 
       
        //copy memory from ram to cuda
        cudaMemcpy(f_in, frame.data, sizeof(uchar)*dHeight*dWidth*3, cudaMemcpyHostToDevice ); 
   
        //multiply the matrices in cuda
        dim3 threadsPerBlock(16,16);
        dim3 numBlocks(ceil(dWidth/(float)16),ceil(dHeight/(float)16));
        histogram_rgbtohsv<<<numBlocks,threadsPerBlock>>>(f_out,f_outi,f_in,dHeight,dWidth);
        /*kernel calls are asynchronous. Hence the checkCudaError() function will execute before the kernel finished.
        In order to tell wait till the kernel is over we use cudaDeviceSynchronize() before checkCudaError()
        In previous error checking examples it should be corrected as this*/
        cudaDeviceSynchronize(); 
           
        //copy the answer back from cuda to ram
        cudaMemcpy(hsvDataPtr, f_out, sizeof(uchar)*dHeight*dWidth*3, cudaMemcpyDeviceToHost ); 
		cudaMemcpy(grayDataPtr, f_outi, sizeof(uchar)*dHeight*dWidth, cudaMemcpyDeviceToHost ); 

        //free the cuda memory
        cudaFree(f_in);
        cudaFree(f_out); 
		cudaFree(f_outi); 
   
        grayDataPtr = histwqualize(grayDataPtr,dHeight,dWidth);
        //grayImg.data = grayDataPtr;
	
        //pointers for memory allocation in cudaa
        uchar *f_in1;
		uchar *f_ini;
        uchar *f_out1;
       
        //allocate memory in cuda
        cudaMalloc((void **)&f_in1,  sizeof(uchar)*dHeight*dWidth*3 ); 
        cudaMalloc((void **)&f_ini,  sizeof(uchar)*dHeight*dWidth );
		cudaMalloc((void **)&f_out1, sizeof(uchar)*dHeight*dWidth*3 ); 
       
        //copy memory from ram to cuda
        cudaMemcpy(f_in1, hsvDataPtr, sizeof(uchar)*dHeight*dWidth*3, cudaMemcpyHostToDevice ); 
		cudaMemcpy(f_ini, grayDataPtr, sizeof(uchar)*dHeight*dWidth, cudaMemcpyHostToDevice ); 
		
        histogram_hsvtorgb<<<numBlocks,threadsPerBlock>>>(f_out1,f_in1,f_ini,dHeight,dWidth);
        /*kernel calls are asynchronous. Hence the checkCudaError() function will execute before the kernel finished.
        In order to tell wait till the kernel is over we use cudaDeviceSynchronize() before checkCudaError()
        In previous error checking examples it should be corrected as this*/
        cudaDeviceSynchronize(); checkCudaError();
       
        //copy the answer back from cuda to ram
        cudaMemcpy(frame.data, f_out1, sizeof(uchar)*dHeight*dWidth*3, cudaMemcpyDeviceToHost ); 

        //free the cuda memory
        cudaFree(f_in1); 
		cudaFree(f_ini);      
	    cudaFree(f_out1); 
 
		//oVideoWriter.write(frame);
		//imshow("Output Video", frame);

    }

	//end measuring time
		
		
		cudaEventCreate(&stop);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedtime,start,stop);
		cout<< "time  "<<elapsedtime/(float)1000<<"\n";
	
	
    return 0;
} 
