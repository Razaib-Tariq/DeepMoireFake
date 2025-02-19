import cv2
import os

def blur_images_with_nested_folders(input_dir, output_dir, blur_ksize=(7, 7)):
    
    os.makedirs(output_dir, exist_ok=True)
    
  
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(root, file)
                
            
                relative_path = os.path.relpath(root, input_dir)
                output_folder_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder_path, exist_ok=True)  
                
                output_image_path = os.path.join(output_folder_path, file)
                
                image = cv2.imread(input_image_path)
                if image is None:
                    print(f"cannot load the image: {input_image_path}")
                    continue
                
                blurred_image = cv2.GaussianBlur(image, blur_ksize, 0)
                
              
                cv2.imwrite(output_image_path, blurred_image)
                print(f"Processed and saved: {output_image_path}")


input_dir = "//"  
output_dir = "//"  
blur_ksize = (7, 7) 
blur_images_with_nested_folders(input_dir, output_dir, blur_ksize)
