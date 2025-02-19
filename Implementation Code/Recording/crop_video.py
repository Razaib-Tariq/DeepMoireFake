import os
import cv2

def crop_video(input_path, x, y, width, height, output_path):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Crop the frame
            crop_frame = frame[y:y+height, x:x+width]
            
            # Write the cropped frame
            out.write(crop_frame)
        else:
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def crop_videos_in_folder(folder_path, x, y, width, height, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  # Check for video files
            video_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f'cropped_{filename}')
            crop_video(video_path, x, y, width, height, output_path)
            print(f"Cropped video saved to: {output_path}")

# Example usage
folder_path = 'FF++/NeuralTextures/NeuralTextures-Real/'
output_folder = 'FF/++/NeuralTextures/NeuralTextures-Real/cropped_videos'
#x, y = 100, 25  # Starting x and y coordinates of the crop area ## for DFD
#width, height = 1715, 960  # Width and height of the crop area ## for DFD

x, y = 115, 25  # Starting x and y coordinates of the crop area ## for DFDC
width, height = 1715, 950  # Width and height of the crop area ## for DFDC

#x, y = 115, 25  # Starting x and y coordinates of the crop area for CelebDF
#width, height = 1715, 960  # Width and height of the crop area ## for CelebDF


crop_videos_in_folder(folder_path, x, y, width, height, output_folder)

