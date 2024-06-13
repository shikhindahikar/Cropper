#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <memory>


#define W_BUFF 1920
#define H_BUFF 1080

// crops a portion of the input image using the top left corner coordinates and the width and height of the crop
// The width and height will always be in uniform factors of 3840x2160 resolution
__global__ void cudacrop(int offsetX, int offsetY, int outWidth, int outHeight, uint8_t* input, uint8_t* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int framesizeblocks = (outWidth * outHeight) ;
    for (int i = index; i < framesizeblocks; i += stride) {
        // altering this will mess with the arrangement of the pixels
        int lineno = (i << 1) / outWidth;
        int column = (i << 1) % outWidth;
        int block = (lineno * outWidth + column);
        // altering this will mess with colours
        if (lineno + offsetY < H_BUFF && column + offsetX < W_BUFF) {
            int inblock = ((lineno + offsetY) * W_BUFF + (column + offsetX));
            output[block] = input[inblock];
            output[block + 1] = input[inblock + 1];
        }
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
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <video file>\n", argv[0]);
        return 1;
    }

    int cropWidth = 960;
    int cropHeight = 540;

    int x = 0, y = 0;

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    std::string video = argv[1];

    // ffmpeg command
    std::string ffmpeg_path = "ffmpeg";
	std::string command = ffmpeg_path + " -i " + video + " -f image2pipe -pix_fmt uyvy422 -vcodec rawvideo -";
	std::FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
        fprintf(stderr, "Failed to open pipe\n");
        SDL_Quit();
        return 1;
	}

    // Create a window
    SDL_Window *win = SDL_CreateWindow("Video Display", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, cropWidth, cropHeight, SDL_WINDOW_SHOWN);
    if (!win) {
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        pclose(pipe);
        SDL_Quit();
        return 1;
    }

    // Create a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        pclose(pipe);
        SDL_Quit();
        return 1;
    }

    // Create a texture for the raw video frame
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_UYVY, SDL_TEXTUREACCESS_STREAMING, cropWidth, cropHeight);
    if (!texture) {
        fprintf(stderr, "SDL_CreateTexture Error: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        pclose(pipe);
        SDL_Quit();
        return 1;
    }

    int quit = 0;
    SDL_Event e;
    void* raw_image = malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    if (!raw_image) {
        fprintf(stderr, "Failed to allocate memory\n");
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        pclose(pipe);
        SDL_Quit();
        return 1;
    }

    // CUDA init
    uint8_t *d_input, *d_output, *d_crop;
    cudaMalloc(&d_input, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    cudaMalloc(&d_output, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    uint8_t *output = (uint8_t*)malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    if (!output) {
        fprintf(stderr, "Failed to allocate memory\n");
        free(raw_image);
        free(output);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        pclose(pipe);
        SDL_Quit();
        return 1;
    }
    while (!quit && fread(raw_image, H_BUFF * W_BUFF * 2, 1, pipe) == 1) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT || e.type == SDL_KEYDOWN) {
                quit = 1;
            }
        }

        // CUDA stuff here
        cudaMemcpy(d_input, raw_image, H_BUFF * W_BUFF * 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_crop, cropWidth * cropHeight * 2 * sizeof(uint8_t));
        cudacrop<<<960, 256>>>(x, y, cropWidth, cropHeight, d_input, d_crop);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in cropping: %s\n", cudaGetErrorString(err));
            break;
        }
        // cudalinearscale<<<960, 256>>>(cropWidth, cropHeight, cropWidth * 2, W_BUFF, H_BUFF, W_BUFF * 2, d_crop, d_output);
        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) {
            fprintf(stderr, "CUDA error scaling: %s\n", cudaGetErrorString(err2));
            break;
        }
        cudaMemcpy(output, d_crop, cropWidth * cropHeight * 2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaFree(d_crop);
        // Update the texture with the new frame data
        // SDL_UpdateTexture(texture, NULL, raw_image, W_BUFF * 2);
        SDL_UpdateTexture(texture, NULL, output, cropWidth * 2);

        // Clear the renderer
        SDL_RenderClear(renderer);

        // Copy the texture to the renderer
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Present the renderer
        SDL_RenderPresent(renderer);
    }

    // Clean up
    free(raw_image);
    free(output);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    pclose(pipe);
    SDL_Quit();

    return 0;
}
