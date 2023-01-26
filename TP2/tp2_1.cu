// tp2_1.cu
//
// This program performs a convolution using the CUDA kernel "conv".
// The convolution itself is rather straightforward: we perform it on the 
// sub-region x => [1, width_max-2] and y => [1, height_max-2], thus the
// boundaries of the image (first/last lines and columns) are not processed.
//
// Better case performance (5 executions) (command: nvcc -Xptxas -O[0-3])
// - compiled with -O0: 163 Gpixel/s (4K image, Jetson TX2 v1)
// - compiled with -O3: 175 Gpixel/s (4K image, Jetson TX2 v1)
//
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
# define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#include <time.h>

void cuda_error(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define cuda_error_check(err) (cuda_error(err, __FILE__, __LINE__))

__device__ int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

__global__
void conv(uint8_t* img_in, uint8_t* img_out, int width, int height)
{
	float const kernel[9] = {
		-1.f, 0.f, 1.f,
    -1.f, 0.f, 1.f,
		-1.f, 0.f, 1.f
	};
	
	// Compute row and column numbers of the pixel's image we're targeting.
  // Each thread is translated of one unit, i.e. idx: [0, max-1] => [1, max]
  // to avoid the 0 row/column.
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// Then, we take care of computing everything from 1 to max-2 
	if (row < height-1 && col < width-1)
	{
		// compute pixel of index (row, col)
		float con = img_in[(row-1)*width+col-1] * kernel[0] + 
                img_in[(row-1)*width+col  ] * kernel[1] +
                img_in[(row-1)*width+col+1] * kernel[2] +
                img_in[(row  )*width+col-1] * kernel[3] +
                img_in[(row  )*width+col  ] * kernel[4] +
                img_in[(row  )*width+col+1] * kernel[5] +
                img_in[(row+1)*width+col-1] * kernel[6] +
                img_in[(row+1)*width+col  ] * kernel[7] +
                img_in[(row+1)*width+col+1] * kernel[8];
		img_out[row*width+col] = clamp((int)con, 0, 255);
	}
}

int main(int argc, char* argv[])
{
	char const *const filename = "image.jpg";
	int width = 0, height = 0, nchannels = 0;
	// request to convert image to gray
	int const desired_channels = 1;

	// load the image
	uint8_t *image_data = stbi_load(filename, &width, &height, 
		                              &nchannels, desired_channels);

	// check for errors
	if (!image_data || !width || !height || !nchannels) {
		printf("Error loading image %s\n" , filename);
		return -1;
	}

	int const nb_bytes = width*height*desired_channels*sizeof(uint8_t);
	uint8_t* img_gpu_in  = NULL;
	uint8_t* img_gpu_out = NULL;
	cuda_error_check(cudaMalloc((void**)&img_gpu_in, nb_bytes));
	cuda_error_check(cudaMalloc((void**)&img_gpu_out, nb_bytes));
	cuda_error_check(cudaMemcpy(img_gpu_in, image_data, nb_bytes, cudaMemcpyHostToDevice));

	// invoke blocks of size 16x16
	dim3 const blockSize(16, 16, 1);
	dim3 const gridSize(ceil(width/16.0f), ceil(height/16.0f), 1);

	double t_moy = 0.;
	clock_t t;

	for (unsigned i = 0; i < 100; i++) {
		t = clock();
    // invoke blocks of size 16x16 
		conv<<<gridSize, blockSize>>>(img_gpu_in, img_gpu_out, width, height);
		t = clock() - t;
		t_moy = t_moy + t;
  }

	t_moy = t_moy/100.0;
	double const elapsed_time_seconds = ((double)t_moy)/CLOCKS_PER_SEC;
  // measure the performance
	printf("Elapsed time : %lf seconds, pixel/second : %lf\n", elapsed_time_seconds, width*height/elapsed_time_seconds);

	uint8_t * filtered_image = (uint8_t*)malloc(nb_bytes);
  cuda_error_check(cudaMemcpy(filtered_image, img_gpu_out, nb_bytes, cudaMemcpyDeviceToHost));

	char const *const outfilename = "image_convoluted.png";
	int const stride = width * 1;

	// save the image
	if (!stbi_write_png(outfilename, width, height, 1, filtered_image, stride)) {
		// use the image data
		// release the image memory buffer
    cuda_error_check(cudaFree(img_gpu_in));
  	cuda_error_check(cudaFree(img_gpu_out));
		free(image_data);
    free(filtered_image);
		return 0;
	}
	return -1;
}

