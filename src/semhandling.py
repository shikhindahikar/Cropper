import numpy as np
import cv2
import posix_ipc
from multiprocessing import shared_memory

shm_name = 'video_frames'
frame_width = 1920
frame_height = 1080
frame_size = (frame_height, frame_width, 2)  # UYVY422 has 2 bytes per pixel

# Open the shared memory
existing_shm = shared_memory.SharedMemory(name=shm_name)
frame_array = np.ndarray(frame_size, dtype=np.uint8, buffer=existing_shm.buf)

sem_producer = posix_ipc.Semaphore("/sem_producer")
sem_consumer = posix_ipc.Semaphore("/sem_consumer")

while True:
    sem_consumer.acquire()

    # Convert UYVY422 to RGB for display
    frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_YUV2BGR_UYVY)
    
    # cv2.imshow('Frame', frame_rgb)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    sem_producer.release()

existing_shm.close()
cv2.destroyAllWindows()
