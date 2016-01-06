#include "opencv2/opencv.hpp"
#include <stdint.h>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

//global variables

uchar* grayDataPtr;
uchar* hsvDataPtr;
uchar *arr;
uchar  *arr1;



//this function parallelizable 
uchar * histwqualize(unsigned char * image, int image_rows, int image_cols)
{

	int i, y, x;
	int histogram[256];
	// Generate the histogram
	for (int i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}



	for (y = 0; y < image_rows; y=y+20)
		for (x = 0; x < image_cols; x=x+20)
			histogram[(int)image[y*image_cols + x]]++;

	



	// Caluculate the size of image
	int size = image_rows * image_cols/400;
	float alpha = (float) 255.0 / size;

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



	for (y = 0; y < image_rows; y++){
		for (x = 0; x < image_cols; x++){
			arr1[y*image_cols + x] = Sk[(int)image[y*image_cols + x]];
			//printf("%d ", new_image[y][x]);
		}
		//printf("\n");
	}

	

	return arr1;
}




int main(int argc, char **argv)
{
	VideoCapture cap("lowcon.mp4"); // Enter the video path here
  	clock_t begin = clock();

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	int dWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	int dHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video


	//allocating memory for matrix
	arr1 = new uchar[dWidth*dHeight];
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
	double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

	cout << "Frame per seconds : " << fps << endl;

	Mat hsvImg, frame1;

	bool bSuccess1 = cap.read(hsvImg);
	bool bSuccess2 = cap.read(frame1);
	int count = 0;
	while (1)
	{
		
		Mat frame;
		Mat grayImg;
		int countblack = 0;

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}

		
	//	imshow("Input Video", frame);

		float fR, fG, fB;
		float fH, fS, fV;
		const float FLOAT_TO_BYTE = 255.0f;
		const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;

		for (int y = 0; y < dHeight; y++){
			for (int x = 0; x < dWidth; x++){

				// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
				uchar bB = frame.data[3 * (dWidth*y + x)];	// Blue component
				uchar bG = frame.data[3 * (dWidth*y + x) + 1];	// Green component
				uchar bR = frame.data[3 * (dWidth*y + x) + 2];	// Red component

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
				fV = fMax;				// Value (Brightness).
				if (iMax != 0) {			// Make sure it's not pure black.
					fS = fDelta / fMax + 0.00001;		// Saturation.
					float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta + 0.00001);	// Make the Hues between 0.0 to 1.0 instead of 6.0
					if (iMax == bR) {		// between yellow and magenta.
						fH = (fG - fB) * ANGLE_TO_UNIT;
					}
					else if (iMax == bG) {		// between cyan and yellow.
						fH = (2.0f / 6.0f) + (fB - fR) * ANGLE_TO_UNIT;
					}
					else {				// between magenta and cyan.
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
					fH = 0;	// undefined hue
					countblack++;
				}

				// Convert from floats to 8-bit integers.
				uchar bH = (uchar)(0.5f + fH * 255.0f);
				uchar bS = (uchar)(0.5f + fS * 255.0f);
				uchar bV = (uchar)(0.5f + fV * 255.0f);

				// Clip the values to make sure it fits within the 8bits.
				

				// Set the HSV pixel components.
				hsvDataPtr[3 * (dWidth*y + x)] = bH;		// H component
				hsvDataPtr[3 * (dWidth*y + x) + 1] = bS;	// S component
				grayDataPtr[(dWidth*y + x)] = intensity;	// V component
				
			}

		}

	

		if (countblack< ((int)dWidth*dHeight*0.8))grayDataPtr = histwqualize(grayDataPtr, dHeight, dWidth);


		for (int i = 0; i < dHeight; i++){
			for (int j = 0; j < dWidth; j++){


				// Get the HSV pixel components
				uchar bH = hsvDataPtr[3 * (dWidth*i + j)];// H component
				uchar bS = hsvDataPtr[3 * (dWidth*i + j) + 1];	// S component
				uchar bV = grayDataPtr[(dWidth*i + j)];	// V component

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
					if (fH >= 1.0f)
						fH = 0.0f;

					fH *= 6.0;			// sector 0 to 5
					fI = floor(fH);		// integer part of h (0,1,2,3,4,5 or 6)
					iI = (int)fH;			//		"		"		"		"
					fF = fH - fI;			// factorial part of h (0 to 1)

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
					default:		// case 5 (or 6):
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
				if (bR > 255)
					bR = 255;
				if (bR < 0)
					bR = 0;
				if (bG > 255)
					bG = 255;
				if (bG < 0)
					bG = 0;
				if (bB > 255)
					bB = 255;
				if (bB < 0)
					bB = 0;

				// Set the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
				frame.data[3 * (dWidth*i + j)] = bB;		// B component
				frame.data[3 * (dWidth*i + j) + 1] = bG;		// G component
				frame.data[3 * (dWidth*i + j) + 2] = bR;		// R component



			}
		}


		//oVideoWriter.write(frame);
	//	imshow("Output Video", frame);


		if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}


		



	}


	clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "time " << (double)elapsed_secs << endl;


	return 0;
}
