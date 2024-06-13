#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <memory>
// #include <nats/nats.h>
#include <semaphore.h>


#define W_BUFF 1920
#define H_BUFF 1080

#define NATS_SERVER "nats://localhost:4222"

// Info to get from NATS
int cropWidth = 1280;
int cropHeight = 720;

const char* sem_producer_name = "sem_producer";
const char* sem_consumer_name = "sem_consumer";

int x = 0, y = 0;

// crops a portion of the input image using the top left corner coordinates and the width and height of the crop
// The width and height will always be in uniform factors of 3840x2160 resolution
__global__ void cudacrop(int offsetX, int offsetY, int outWidth, int outHeight, uint8_t* input, uint8_t* output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int framesizeblocks = (outWidth * outHeight) >> 1;

    for (int i = index; i < framesizeblocks; i += stride) {
        // Calculate the line number and column for the current pixel
        int lineno = (i << 1) / outWidth;
        int column = (i << 1) % outWidth;

        // Calculate the linear index for the output buffer
        int outIndex = lineno * outWidth + column;

        // Calculate the corresponding linear index for the input buffer
        int inRow = lineno + offsetY;
        int inCol = column + offsetX;
        int inIndex = inRow * W_BUFF + inCol;

        // Copy the pixel data from input to output buffer
        if (inRow < H_BUFF && inCol < W_BUFF) {
            // Copy the pixel data from input to output buffer
            ((uint32_t*)output)[outIndex >> 1] = ((uint32_t*)input)[inIndex >> 1];
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

// void call_back(natsConnection *conn, natsSubscription *sub, natsMsg *msg, void *closure) {
//     printf("Received message: %s\n", natsMsg_GetData(msg));
//     // parse the message and update the crop coordinates
//     sscanf(natsMsg_GetData(msg), "%d %d %d %d", &x, &y, &cropWidth, &cropHeight);
//     natsMsg_Destroy(msg);
// }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <video file>\n", argv[0]);
        return 1;
    }

    // Initialize NATS
    // natsConnection *conn = NULL;
    // natsConnection_ConnectTo(&conn, NATS_SERVER);
    // if (conn == NULL) {
    //     fprintf(stderr, "Failed to connect to NATS server\n");
    //     return 1;
    // }

    // natsSubscription *sub = NULL;
    // natsConnection_Subscribe(&sub, conn, "digital_PTZ.*", call_back, NULL);

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

    // Create a shared memory object
    const char* shm_name = "video_frames";
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return -1;
    }

    // Set the size of the shared memory object
    if (ftruncate(shm_fd, W_BUFF * H_BUFF * 2) == -1) {
        perror("ftruncate");
        return -1;
    }

    // Map the shared memory object
    void* shm_ptr = mmap(0, W_BUFF * H_BUFF * 2, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("mmap");
        return -1;
    }

    // Create a window
    SDL_Window *win = SDL_CreateWindow("Video Display", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W_BUFF, H_BUFF, SDL_WINDOW_SHOWN);
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
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_UYVY, SDL_TEXTUREACCESS_STREAMING, W_BUFF, H_BUFF);
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
    // void* raw_image = malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    // if (!raw_image) {
    //     fprintf(stderr, "Failed to allocate memory\n");
    //     SDL_DestroyTexture(texture);
    //     SDL_DestroyRenderer(renderer);
    //     SDL_DestroyWindow(win);
    //     pclose(pipe);
    //     SDL_Quit();
    //     return 1;
    // }

    // CUDA init
    uint8_t *d_input, *d_output, *d_crop;
    cudaMalloc(&d_input, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    cudaMalloc(&d_output, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    uint8_t *output = (uint8_t*)malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
    if (!output) {
        fprintf(stderr, "Failed to allocate memory\n");
        // free(raw_image);
        free(output);
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        pclose(pipe);
        SDL_Quit();
        return 1;
    }

    sem_t* sem_producer = sem_open(sem_producer_name, O_CREAT, 0666, 1);
    sem_t* sem_consumer = sem_open(sem_consumer_name, O_CREAT, 0666, 0);

    if (sem_producer == SEM_FAILED || sem_consumer == SEM_FAILED) {
        perror("sem_open");
        return -1;
    }

    while (!quit && fread(shm_ptr, H_BUFF * W_BUFF * 2, 1, pipe) == 1) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT || e.type == SDL_KEYDOWN) {
                quit = 1;
            }
        }

        sem_wait(sem_producer);

        // CUDA stuff here
        cudaMemcpy(d_input, shm_ptr, H_BUFF * W_BUFF * 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_crop, cropWidth * cropHeight * 2 * sizeof(uint8_t));
        cudacrop<<<960, 256>>>(x, y, cropWidth, cropHeight, d_input, d_crop);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in cropping: %s\n", cudaGetErrorString(err));
            break;
        }
        cudalinearscale<<<960, 256>>>(cropWidth, cropHeight, cropWidth, W_BUFF, H_BUFF, W_BUFF, d_crop, d_output);
        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) {
            fprintf(stderr, "CUDA error scaling: %s\n", cudaGetErrorString(err2));
            break;
        }
        cudaMemcpy(output, d_output, W_BUFF * H_BUFF * 2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaFree(d_crop);
        // Update the texture with the new frame data
        // SDL_UpdateTexture(texture, NULL, raw_image, W_BUFF * 2);
        SDL_UpdateTexture(texture, NULL, output, W_BUFF * 2);

        // Clear the renderer
        SDL_RenderClear(renderer);

        // Copy the texture to the renderer
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Present the renderer
        SDL_RenderPresent(renderer);

        sem_post(sem_consumer);
    }

    // Clean up
    // free(raw_image);
    free(output);
    // SDL closing
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
    // closing the shared memory stuff
    munmap(shm_ptr, W_BUFF * H_BUFF * 2);
    close(shm_fd);
    shm_unlink(shm_name);
    // closing the semaphores
    sem_close(sem_producer);
    sem_close(sem_consumer);
    sem_unlink(sem_producer_name);
    sem_unlink(sem_consumer_name);
    // closing the pipe
    pclose(pipe);

    return 0;
}
