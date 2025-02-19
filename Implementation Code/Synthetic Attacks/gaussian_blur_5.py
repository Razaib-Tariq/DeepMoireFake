import cv2
import os

def blur_images_with_nested_folders(input_dir, output_dir, blur_ksize=(5, 5)):
    # 출력 경로가 존재하지 않으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 디렉토리 탐색
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 이미지 파일만 처리
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(root, file)
                
                # 출력 경로 설정 (입력 폴더의 구조를 그대로 유지)
                relative_path = os.path.relpath(root, input_dir)
                output_folder_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder_path, exist_ok=True)  # 폴더 생성
                
                # 출력 이미지 경로 설정
                output_image_path = os.path.join(output_folder_path, file)
                
                # 이미지 블러링 처리
                image = cv2.imread(input_image_path)
                if image is None:
                    print(f"이미지를 불러올 수 없습니다: {input_image_path}")
                    continue
                
                blurred_image = cv2.GaussianBlur(image, blur_ksize, 0)
                
                # 결과 저장
                cv2.imwrite(output_image_path, blurred_image)
                print(f"Processed and saved: {output_image_path}")

# 경로 설정
input_dir = "/media/NAS/DATASET/Moire-Pattern/OG_Subset/FF++/"  # 입력 폴더 경로 (예: deepfakes/f2f/fs/nn)
output_dir = "/media/NAS/DATASET/Moire-Pattern/Gussian_blur_5/"  # 출력 폴더 경로
blur_ksize = (5, 5)  # Gaussian Blur 커널 크기
blur_images_with_nested_folders(input_dir, output_dir, blur_ksize)
