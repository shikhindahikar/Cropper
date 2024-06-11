#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <memory>


#define W_BUFF 1920
#define H_BUFF 1080

// crops a portion of the input image using the top left corner coordinates and the width and height of the crop
__global__ void cudacrop(int x, int y, int cropwidth, int cropheight, int inputpitch, int outputpitch, uint8_t* input, uint8_t* output) {
    // Check if crop is within the input frame of UYVY format
    if (x + cropwidth > W_BUFF || y + cropheight > H_BUFF) {
        return;
    }

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int framesizeblocks = cropwidth * cropheight;

    for (int i = index; i < framesizeblocks; i += stride) {
        int lineno = i / cropwidth;
        int column = i % cropwidth;
        int inputIndex = ((lineno + y) * inputpitch + (column + x) * 2);
        int outputIndex = (lineno * outputpitch + column * 2);
        
        output[outputIndex] = input[inputIndex];
        output[outputIndex + 1] = input[inputIndex + 1];
    }
}


// Scales the input image to the output image
__global__ void cudalinearscale(int inwidth, int inheight, int inpitch, int outwidth, int outheight, int outpitch, uint8_t *input, uint8_t *output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int framesizeblocks = (outwidth * outheight) >> 1;
	for(int i = index; i < framesizeblocks; i += stride) {
		int lineno = (i << 1) / outwidth;
		int column = (i << 1) % outwidth;
		float inlineno = (lineno * inheight);
		inlineno /= (float)outheight;
		float incolno = (column * inwidth);
		incolno /= (float)outwidth;
		int block = (lineno * outpitch + column) * 2;

		int topleft = (((int)floorf(inlineno) * inpitch + (int)floorf(incolno))) << 1;
		int topright = (((int)floorf(inlineno) * inpitch + (int)ceilf(incolno))) << 1;
		int botleft = (((int)ceilf(inlineno) * inpitch + (int)floorf(incolno))) << 1;
		int botright = (((int)ceilf(inlineno) * inpitch + (int)ceilf(incolno))) << 1;
		//output[(i<<2)] = 128;
		//output[(i<<2)+2] = 128;
		//output[(i<<2)+1] = (i<<1); 
		//output[(i<<2)+3] = 1-((i<<1)+1);
		//output[(i<<2)+3] = 0;
		float y, u, v;
		// Y 1
		if(topright == topleft) {
			if(botleft == topleft) {
				y = input[topleft + 1];
				u = input[topleft & 0xFFFFFFFC];
				v = input[(topleft & 0xFFFFFFFC) + 2];

			} else {
				y = ((float)input[topleft + 1]) * (1.0 - (inlineno - floorf(inlineno)))
					+ ((float)input[botleft + 1]) * (1.0 - (ceilf(inlineno) - inlineno));
				u = ((float)input[topleft & 0xFFFFFFFC]) * (1.0 - (inlineno - floorf(inlineno)))
					+ ((float)input[botleft & 0xFFFFFFFC]) * (1.0 - (ceilf(inlineno) - inlineno));
				v = ((float)input[(topleft & 0xFFFFFFFC) + 2]) * (1.0 - (inlineno - floorf(inlineno)))
					+ ((float)input[(botleft & 0xFFFFFFFC) + 2]) * (1.0 - (ceilf(inlineno) - inlineno));
			}
		} else {
			if(botleft == topleft) {
				y = (((float)input[topleft + 1]) * (1.0 - (incolno - floorf(incolno))))
					+ (((float)input[topright + 1]) * (1.0 - (ceilf(incolno) - incolno)));
				u = (((float)input[topleft & 0xFFFFFFFC]) * (1.0 - (incolno - floorf(incolno))))
					+ (((float)input[topright & 0xFFFFFFFC]) * (1.0 - (ceilf(incolno) - incolno)));
				u = (((float)input[(topleft & 0xFFFFFFFC) + 2]) * (1.0 - (incolno - floorf(incolno))))
					+ (((float)input[(topright & 0xFFFFFFFC) + 2]) * (1.0 - (ceilf(incolno) - incolno)));
			} else {
				y = (((float)input[topleft + 1]) * (1.0 - (incolno - floorf(incolno))) + 
						((float)input[topright + 1]) * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (inlineno - floorf(inlineno)))
					+	(((float)input[botleft + 1]) * (1.0 - (incolno - floorf(incolno))) + 
							((float)input[botright + 1]) * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (ceilf(inlineno) - inlineno));
				u =	((float)input[topleft & 0xFFFFFFFC] * (1.0 - (incolno - floorf(incolno))) + 
						(float)input[topright & 0xFFFFFFFC] * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (inlineno - floorf(inlineno)))
					+	((float)input[botleft & 0xFFFFFFFC] * (1.0 - (incolno - floorf(incolno))) + 
							(float)input[botright & 0xFFFFFFFC] * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (ceilf(inlineno) - inlineno));
				v = ((float)input[(topleft & 0xFFFFFFFC) + 2] * (1.0 - (incolno - floorf(incolno))) + 
						(float)input[(topright & 0xFFFFFFFC) + 2] * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (inlineno - floorf(inlineno)))
					+	((float)input[(botleft & 0xFFFFFFFC) + 2] * (1.0 - (incolno - floorf(incolno))) + 
							(float)input[(botright & 0xFFFFFFFC) + 2] * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (ceilf(inlineno) - inlineno));
			}
		}
		if(y < 0.0) output[block + 1] = 0;
		else if(y > 255.0) output[block + 1] = 255;
		else output[block + 1] = y;
		if(u < 0.0) output[block] = 0;
		else if(u > 255.0) output[block] = 255;
		else output[block] = u;
		if(v < 0.0) output[block+2] = 0;
		else if(v > 255.0) output[block+2] = 255;
		else output[block+2] = v;

		output[block] = input[(topleft & 0xFFFFFFFC)];
		output[block+2] = input[(topleft & 0xFFFFFFFC)+2];

		// Y 2
		incolno = (((float)column + 1.0) * (float)inwidth);
		incolno /= (float)outwidth;
		topleft = (((int)floorf(inlineno) * inpitch + (int)floorf(incolno))) << 1;
		topright = (((int)floorf(inlineno) * inpitch + (int)ceilf(incolno))) << 1;
		botleft = (((int)ceilf(inlineno) * inpitch + (int)floorf(incolno))) << 1;
		botright = (((int)ceilf(inlineno) * inpitch + (int)ceilf(incolno))) << 1;
		if(topright == topleft) {
			if(botleft == topleft) {
				y = input[topleft + 1];
			} else {
				y = ((float)input[topleft + 1]) * (1.0 - (inlineno - floorf(inlineno)))
					+ ((float)input[botleft + 1]) * (1.0 - (ceilf(inlineno) - inlineno));
			}
		} else {
			if(botleft == topleft) {
				y = (((float)input[topleft + 1]) * (1.0 - (incolno - floorf(incolno))))
					+ (((float)input[topright + 1]) * (1.0 - (ceilf(incolno) - incolno)));
			} else {
				y = (((float)input[topleft + 1]) * (1.0 - (incolno - floorf(incolno))) + 
						((float)input[topright + 1]) * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (inlineno - floorf(inlineno)))
					+	(((float)input[botleft + 1]) * (1.0 - (incolno - floorf(incolno))) + 
							((float)input[botright + 1]) * (1.0 - (ceilf(incolno) - incolno))) * (1.0 - (ceilf(inlineno) - inlineno));
			}
		}

		if(y < 0.0) output[block+3] = 0;
		else if(y > 255.0) output[block+3] = 255;
		else output[block+3] = y;
		//output[block + 3] = input[topleft + 1];
	}
}

int main(int argc, char* argv[]) {
    int cropWidth = 960;
    int cropHeight = 540;
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    std::string video = argv[1];

    // ffmpeg shit
    std::string ffmpeg_path = "ffmpeg";

    // command to run ffmpeg
	std::string command = ffmpeg_path + " -i " + video + " -f image2pipe -pix_fmt uyvy422 -vcodec rawvideo -";
	std::FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("Failed to open pipe");
	}

	void* raw_image = malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));

    // Create a window
    SDL_Window *win = SDL_CreateWindow("Video Display", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W_BUFF, H_BUFF, SDL_WINDOW_SHOWN);
    if (win == NULL) {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Create a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
        SDL_DestroyWindow(win);
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Create a texture for the raw video frame
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_UYVY, SDL_TEXTUREACCESS_STREAMING, W_BUFF, H_BUFF);
    if (texture == NULL) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        fprintf(stderr, "SDL_CreateTexture Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // CUDA init
    uint8_t* d_input;
    uint8_t* d_crop_input;
    uint8_t* d_output;
    uint8_t* output;

    int quit = 0;
    SDL_Event e;
    while (!quit && fread(raw_image, H_BUFF * W_BUFF * 2, 1, pipe)) {
        while (SDL_PollEvent(&e) != 0) {
            // User requests quit
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            else if (e.type == SDL_KEYDOWN) {
                // User presses a key
                quit = true;
            }
        }

        // Do CUDA shit here
        cudaMalloc((void**)d_input, H_BUFF * W_BUFF * sizeof(uint8_t) * 2);
        cudaMalloc((void**)d_output, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
        cudaMalloc((void**)d_crop_input, 960 * 540 * 2 * sizeof(uint8_t));
        output = (uint8_t*)malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
        cudaMemcpy(d_input, raw_image, H_BUFF * W_BUFF * sizeof(uint8_t) * 2, cudaMemcpyHostToDevice);
        // Getting a crop from a frame into another frame
        // Define block and grid dimensions for CUDA kernel launch
        int blockSize = 256;
        int numBlocks = (cropWidth * cropHeight + blockSize - 1) / blockSize;
        fprintf(stderr, "Reached here!");
        cudacrop<<<numBlocks, blockSize>>>(0, 0, cropWidth, cropHeight, W_BUFF * 2, cropWidth * 2, d_input, d_crop_input);
        fprintf(stderr, "Passed this");
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // scaling the cropped frame to the original frame size
        cudalinearscale<<<960, 256>>>(cropWidth, cropHeight, cropWidth * 2, W_BUFF, H_BUFF, W_BUFF * 2, d_crop_input, d_output);

        cudaMemcpy(output, d_output, H_BUFF * W_BUFF * 2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        
        // Update the texture with the new frame data
        SDL_UpdateTexture(texture, NULL, output, W_BUFF * 2);

        // Clear the renderer
        SDL_RenderClear(renderer);

        // Copy the texture to the renderer
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Present the renderer
        SDL_RenderPresent(renderer);

        // Delay to simulate frame rate
        SDL_Delay(33); // ~30 FPS

        free(output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_crop_input);
    }

    // Clean up
    free(raw_image);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_crop_input);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}
