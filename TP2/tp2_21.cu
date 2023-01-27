// tp2_21.cu
//
// This program performs a convolution using the CUDA kernel "conv".
// Each thread in the same block shares the same array "_block".
// It doesn't take into account the boundary of each 16x16 region, i.e.
// each block of threads performs the convolution on the 14x14 sub-window
// and thus leaves the first/last line/column (of the region processed by
// the block) dark.
//
// Better case performance (5 executions) (command: nvcc -Xptxas -O[0-3])
// - compiled with -O0: 174 Gpixel/s (4K image, Jetson TX2 v1)
// - compiled with -O3: 161 Gpixel/s (4K image, Jetson TX2 v1)
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
	// This array is shared by all the threads in the same block
    __shared__ float _block[16][16];
	
    // Convolution mask definition
	float const kernel[9] = {
		-1.f, 0.f, 1.f,
        -1.f, 0.f, 1.f,
		-1.f, 0.f, 1.f
	};
	
	// Compute row and column numbers of the pixel's image we're targeting
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread in the same block (y_idx, x_idx in [0,15], see invocation
    // parameters) loads its corresponding image value, thus we have all the
    // necessary values to compute a 2d convolution only on y_idx, x_idx in 
    // [1,14]
	if (row < height && col < width)
	  _block[threadIdx.y][threadIdx.x] = img_in[row*width+col];

	// wait for all the threads to load its value
	__syncthreads();

	// Each thread (except the ones on the boundary region, i.e. all threads 
    // having an idx x or y = 0 or 15) applies the convolution mask (2d conv) 
    // to compute its corresponding pixel. 
    // In this way, we prevent corner cases where we compute a convolution
    // that needs to access out of range values, particularly if:
    //   |*| threadIdx.(y|x)=0
    //   |*| threadIdx.(y|x)=15
    //
	if (row < height && col < width &&
      // Make sure that the accessed row and column of _block are in [0,15]
      // by reducing the convolution window on [1,14].
      threadIdx.y > 0 && threadIdx.x > 0 && threadIdx.y < 15 && threadIdx.x < 15)
	{
		// Compute pixel of index (row, col)
		float con = _block[threadIdx.y-1][threadIdx.x-1] * kernel[0] + 
                    _block[threadIdx.y-1][threadIdx.x  ] * kernel[1] +
                    _block[threadIdx.y-1][threadIdx.x+1] * kernel[2] +
                    _block[threadIdx.y  ][threadIdx.x-1] * kernel[3] +
                    _block[threadIdx.y  ][threadIdx.x  ] * kernel[4] +
                    _block[threadIdx.y  ][threadIdx.x+1] * kernel[5] +
                    _block[threadIdx.y+1][threadIdx.x-1] * kernel[6] +
                    _block[threadIdx.y+1][threadIdx.x  ] * kernel[7] +
                    _block[threadIdx.y+1][threadIdx.x+1] * kernel[8];
		img_out[row*width+col] = clamp((int)con, 0, 255);
	}
	// Otherwise, put a black pixel.
    // As a result, we should see an image "quadrillée" by black rows and 
    // columns as each 16x16 region of the image is processed by a given block.
	else img_out[row*width+col] = 0;
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

