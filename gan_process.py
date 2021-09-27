import albumentations as A
import cv2 
import numpy as np
import torch
import os

from PIL import Image
from facemask_detection.pre_trained_models import get_model as get_classifier
from retinaface.pre_trained_models import get_model as get_detector
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from module import detect_model
from module import edit_model


# 이미지 input 부분
def goGan():
    img_gt = cv2.imread("/home/gan_unmask/GAN_Service/static/temp/01.jpg")
    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    h_gt,w_gt,_1=img_gt.shape
    #croplist
    face_detector = get_detector("resnet50_2020-07-20", max_size=800)
    face_detector.eval()
    with torch.no_grad():
        annotations = face_detector.predict_jsons(img_gt)

    mask_classifier = get_classifier("tf_efficientnet_b0_ns_2020-07-29")
    mask_classifier.eval()
    transform = A.Compose([A.SmallestMaxSize(max_size=256, p=1),
                        A.CenterCrop(height=224, width=224, p=1),
                        A.Normalize(p=1)])
    predictions = []

    with torch.no_grad():
        for annotation in tqdm(annotations):
            x_min, y_min, x_max, y_max = annotation['bbox']

            x_min = np.clip(x_min, 0, x_max)
            y_min = np.clip(y_min, 0, y_max)

            crop = img_gt[y_min:y_max, x_min:x_max]
            crop_transformed = transform(image=crop)['image']
            model_input = torch.from_numpy(np.transpose(crop_transformed, (2, 0, 1)))
            predictions += [mask_classifier(model_input.unsqueeze(0))[0].item()]

    vis_image = img_gt.copy()
    crop_list = []
    coordinate = []
    for prediction_id, annotation in enumerate(annotations):
        is_mask = predictions[prediction_id] > 0.5
        if is_mask:
            color = (255, 0, 0)
            x_min, y_min, x_max, y_max = annotation["bbox"]

            if ((x_min-26)<=0 or (y_min-26)<=0 or (x_max+26)>=w_gt or (y_max+26)>=h_gt) and len(predictions)==1:
                crop_list=[vis_image]
                coordinate=[[0,h_gt,0,w_gt]]
                print('잘걸렀어')
                break

            x_min = np.clip(x_min-26, 0, w_gt-1)
            y_min = np.clip(y_min-26, 0, h_gt-1)
            x_max = np.clip(x_max+26, 0, w_gt-1)
            y_max = np.clip(y_max+26, 0, h_gt-1)

            vis_image_a = vis_image[y_min :y_max , x_min :x_max ]
            coordinate.append([y_min,y_max, x_min, x_max])
            crop_list.append(vis_image_a)

    #last crop position
    hw_list = []
    for x in crop_list:
        a,b,c = x.shape
        hw_list.append((a,b))

    # Tensor로 바꾸기 
    loader_color = transforms.Compose([transforms.ToTensor()])
    loader_gray = transforms.Compose([transforms.ToTensor(),
                                    transforms.Grayscale(num_output_channels=1)])  # 토치 텐서로 변환
    # image 불러오는 함수
    def image_loader_color(image):
        image = cv2.resize(image, (160, 160))
        image = loader_color(image).unsqueeze(0)
        return image

    # Detector weight load
    state_D = torch.load('/home/gan_unmask/GAN_Service/module/detect_1000img_300epoch.pth', map_location='cpu')

    trained_Dmodel = detect_model.Detector() 
    trained_Dmodel.load_state_dict(state_D)

    # Editor weight load
    state_G= torch.load('/home/gan_unmask/GAN_Service/module/model_48_batch16.pth', map_location='cpu')

    trained_Gmodel = edit_model.GatedGenerator()
    trained_Gmodel.load_state_dict(state_G['G'])

    masked = crop_list
    length = len(masked)
    #generated = []

    # range: 0~croped, 파일 이름 따라서 수정 주의 
    for i in range(length):
    
        # Detector 모델에 넣고 리스트에 append
        temp_m = image_loader_color(masked[i])
        temp_b = trained_Dmodel.forward(temp_m)

        # binary 파일 후처리후 저장
        temp_b[temp_b>0.5] = 1.0
        temp_b[temp_b<=0.5] = 0
        img_list  = [i.clone().cpu() for i in temp_b]
        temp_R = torch.stack(img_list, dim=0)
        save_image(temp_R,f'/home/gan_unmask/GAN_Service/img_temp/dtemp{i}.jpg')

        # 다시 불러온후 sharpen
        img1 = cv2.imread(f'/home/gan_unmask/GAN_Service/img_temp/dtemp{i}.jpg')
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        dilation = cv2.dilate(img1, kernel3, iterations=1)
        after_sharpen = cv2.fastNlMeansDenoisingColored(dilation,None,50,50,7,21)

        # sharpen 후처리
        after_sharpen = loader_gray(after_sharpen)
        after_sharpen = torch.stack([after_sharpen])
        
        # Editor 모델에 넣고 후처리
        first_out, second_out = trained_Gmodel(temp_m, after_sharpen)
        first_out_wholeimg = temp_m * (1 - after_sharpen) + first_out * after_sharpen
        second_out_wholeimg = temp_m * (1 - after_sharpen) + second_out * after_sharpen
        img_list = [second_out_wholeimg]
        imgs = torch.stack(img_list, dim=1)

        # 결과 사진 저장 
        imgs = imgs.view(-1, *list(imgs.size())[2:])
        save_img_path = (f'/home/gan_unmask/GAN_Service/img_temp/gtemp{i}.jpg')
        save_image(imgs, save_img_path)

    img_gt = Image.fromarray(img_gt)
    for i in range(length):
        im2 = cv2.imread(f'/home/gan_unmask/GAN_Service/img_temp/gtemp{i}.jpg')
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        im2 = cv2.resize(im2,(hw_list[i][1],hw_list[i][0]))
        im2 = Image.fromarray(im2)
        img_gt.paste(im2,(coordinate[i][2],coordinate[i][0],coordinate[i][3],coordinate[i][1]))
    wholelen = len(os.listdir('/home/gan_unmask/GAN_Service/static/after'))
    img_gt2 = img_gt.copy()
    img_gt.save('/home/gan_unmask/GAN_Service/static/temp/02.jpg')
    img_gt2.save('/home/gan_unmask/GAN_Service/static/after/after_'+f'{wholelen+1}'.zfill(5)+'.jpg')

'''
if __name__ == '__main__':
    goGan()
'''


