#include <SDL2/SDL.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <memory>

#define W_BUFF 1920
#define H_BUFF 1080

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    std::string video = argv[1];

    // ffmpeg shit
    std::string ffmpeg_path = "ffmpeg";

    // command to run ffmpeg
	std::string command = ffmpeg_path + " -i " + video + " -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -";
	std::FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("Failed to open pipe");
	}

	void* raw_image = malloc(H_BUFF * W_BUFF * 3 * sizeof(uint8_t));

    // Create a window
    SDL_Window *win = SDL_CreateWindow("Video Display", 100, 100, W_BUFF, H_BUFF, SDL_WINDOW_SHOWN);
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
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, W_BUFF, H_BUFF);
    if (texture == NULL) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(win);
        fprintf(stderr, "SDL_CreateTexture Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    int quit = 0;
    SDL_Event event;
    while (!quit && fread(raw_image, H_BUFF * W_BUFF * 3, 1, pipe)) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = 1;
            }
        }

        // Update the texture with the new frame data
        SDL_UpdateTexture(texture, NULL, raw_image, W_BUFF * 3);

        // Clear the renderer
        SDL_RenderClear(renderer);

        // Copy the texture to the renderer
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Present the renderer
        SDL_RenderPresent(renderer);

        // Delay to simulate frame rate
        SDL_Delay(33); // ~30 FPS
    }

    // Clean up
    free(raw_image);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();

    return 0;
}
