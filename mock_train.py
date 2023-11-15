import os
import time
import random

def main():
    # Create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Generate a random duration between 1 and 5 seconds
    duration = random.randint(1, 5)
    print(f"Running for {duration} seconds...")

    # Create an empty file named unet_denoise.h5
    with open('unet_denoise.h5', 'w') as file:
        pass

    # Sleep for the duration
    time.sleep(duration)

    print("Execution completed.")

if __name__ == "__main__":
    main()
