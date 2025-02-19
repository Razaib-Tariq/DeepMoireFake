import cv2
import glob
import numpy as np
import os
import re
import time
from auto_record import selenium_script_start
from auto_stop import selenium_script_stop
import threading
from screeninfo import get_monitors

def timed_input(prompt, timeout=10):
    print(prompt, end="", flush=True)
    result = [None]

    def get_input():
        result[0] = input()

    thread = threading.Thread(target=get_input)
    thread.daemon = True  # Set thread as daemon
    thread.start()
    thread.join(timeout)

    if result[0] is None:
        print("\nTime's up! Continuing to next batch.")
        return "yes"

    return result[0]

def save_processed_video(index, video_name, filepath="processed_videos-random.txt"):
    with open(filepath, "a") as file:
        file.write(f"{index},{video_name}\n")


def load_last_index(filepath="processed_videos-random.txt"):
    try:
        with open(filepath, "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                last_index, _ = last_line.split(',')
                return int(last_index) + 1
            return 0
    except FileNotFoundError:
        return 0  # If file not found, start from beginning

def extract_numeric_part(filename):
    # Extract numeric part from the filename (e.g., 'id0_0001' -> 0)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def sort_key(path):
    dirname = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)

    try:
        # Try to convert the directory name to an integer for numeric sorting
        numeric_dirname = int(dirname)
        is_numeric = True
    except ValueError:
        # If conversion fails, use the original string for alphabetical sorting
        numeric_dirname = dirname
        is_numeric = False

    # For non-numeric directories, extract numeric part of the filename for sorting
    numeric_part = extract_numeric_part(filename) if not is_numeric else 0

    # Custom sorting: Sort by whether it's numeric, then by numeric/alphabetical folder name, then by numeric part of filename, and finally by filename
    return (not is_numeric, numeric_dirname, numeric_part, path)

def get_all_video_files(root_folder):
    video_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_files.append(os.path.join(dirpath, filename))

    video_files.sort(key=sort_key)
    return video_files

def opencv_script(driver, URL, start_button, stop_button):
    video_folder = "\Moire\Dataset\-randomvideos"
    video_files = get_all_video_files(video_folder)
    total_videos = len(video_files)
    batch_size = 500

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    window_name = 'Video Player'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_index = load_last_index()
    while start_index < total_videos:
        end_index = min(start_index + batch_size, total_videos)

        for index in range(start_index, end_index):
            video_file = video_files[index]
            print(f"Playing video {index + 1} of {total_videos}. Remaining: {total_videos - index - 1}")

            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Error opening video file {video_file}")
                continue

            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if original_width <= screen_width and original_height <= screen_height:
                x_offset = (screen_width - original_width) // 2
                y_offset = (screen_height - original_height) // 2
            else:
                # If video is larger than the screen, scale it down
                scaling_factor = min(screen_width / original_width, screen_height / original_height)
                scaled_width = int(original_width * scaling_factor)
                scaled_height = int(original_height * scaling_factor)
                x_offset = (screen_width - scaled_width)
                y_offset = (screen_height - scaled_height)

            pre_record_buffer = 1  # Delay before starting recording
            post_playback_buffer = 1  # Delay after stopping playback
            
            play_duration = 18
            start_time = time.time()

            ret, frame = cap.read()
            if ret:
                # Add border to center the frame in the window
                bordered_frame = cv2.copyMakeBorder(frame, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT)
                cv2.imshow(window_name, bordered_frame)
                cv2.waitKey(2000)  # Display the first frame for 2 seconds

            time.sleep(pre_record_buffer)
            print(f"Starting recording for video {index + 1}")
            selenium_script_start(driver, URL, start_button)

            adjusted_play_duration = play_duration - pre_record_buffer - post_playback_buffer

            while ret and (time.time() - start_time) < adjusted_play_duration:
                ret, frame = cap.read()
                if ret:
                    bordered_frame = cv2.copyMakeBorder(frame, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT)
                    cv2.imshow(window_name, bordered_frame)
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
            time.sleep(post_playback_buffer)

            cap.release()
            print(f"Stopping recording for video {index + 1}")
            selenium_script_stop(driver, URL, stop_button)

            if index < total_videos - 1:
                cv2.imshow(window_name, np.zeros((screen_height, screen_width, 3), dtype='uint8'))
                cv2.waitKey(1000)

            save_processed_video(index, video_file)

        if end_index < total_videos:
            response = timed_input("Continue with the next batch of videos? (yes/no): ", 10)
            if response.lower() != 'yes':
                break

        start_index = end_index

    cv2.destroyAllWindows()