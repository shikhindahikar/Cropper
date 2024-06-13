#include <SDL2/SDL.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <memory>
#include <nats/nats.h>
#include <semaphore.h>


#define W_BUFF 1920
#define H_BUFF 1080

#define NATS_SERVER "nats://localhost:4222"