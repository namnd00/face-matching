import os
import cv2
import numpy as np
import torch

from imgaug import augmenters as iaa
from skimage import transform as trans
from shutil import copyfile
from tqdm import *
from skimage import io
from facenet_pytorch import MTCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = MTCNN(keep_all=True, post_process=False, device=device)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


def detect_gender(aligned):
    genderList = ['M', 'F']

    blob = cv2.dnn.blobFromImage(aligned, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()

    return genderList[genderPreds[0].argmax()]


def alignment(cv_img, dst, dst_w, dst_h):
    if dst_w == 96 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
    elif dst_w == 112 and dst_h == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
    elif dst_w == 150 and dst_h == 150:
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif dst_w == 160 and dst_h == 160:
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif dst_w == 224 and dst_h == 224:
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        return None
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    face_img = cv2.warpAffine(cv_img, M, (dst_w, dst_h), borderValue=0.0)
    return face_img


def main():
    count = 0
    total = 0
    for mset in ['train', 'test']:
        if not os.path.exists(f'../datasets/aligned/96x112/{mset}'):
            os.makedirs(f'../datasets/aligned/96x112/{mset}')
        if not os.path.exists(f'../datasets/aligned/112x112/{mset}'):
            os.makedirs(f'../datasets/aligned/112x112/{mset}')
        if not os.path.exists(f'../datasets/aligned/150x150/{mset}'):
            os.makedirs(f'../datasets/aligned/150x150/{mset}')
        if not os.path.exists(f'../datasets/aligned/160x160/{mset}'):
            os.makedirs(f'../datasets/aligned/160x160/{mset}')
        if not os.path.exists(f'../datasets/aligned/224x224/{mset}'):
            os.makedirs(f'../datasets/aligned/224x224/{mset}')

        if not os.path.exists(f'../datasets/unknown'):
            os.makedirs(f'../datasets/unknown')

        if not os.path.exists(f'../datasets/not_aligned'):
            os.makedirs(f'../datasets/not_aligned')

        unknown_file = open(f'../datasets/unknown.txt', 'w')
        not_aligned_file = open(f'../datasets/not_aligned.txt', 'w')

        for rdir, _, files in os.walk(f'../datasets/images/{mset}'):
            for file in tqdm(files):
                img_path = os.path.join(rdir, file)
                try:
                    if '.jpg' not in file:
                        continue
                    image = io.imread(img_path)
                    _, _, landmarks = detector.detect(image, landmarks=True)

                    check = False
                    if landmarks is None:
                        print('[INFO] Step 1: unknown ' + img_path)
                        for sigma in np.linspace(0.0, 3.0, num=11).tolist():
                            seq = iaa.GaussianBlur(sigma)
                            image_aug = seq.augment_image(image)
                            _, _, landmarks = detector.detect(image_aug, landmarks=True)
                            if landmarks is not None:
                                print('[INFO] Sigma:', sigma)
                                for ix, point in enumerate(landmarks):
                                    dst = np.float32(point)
                                    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                    file = "{}.{}.jpg".format(file.split(".jpg")[0], ix + 1)

                                    face_96x112 = alignment(cv_img, dst, 96, 112)

                                    cv2.imwrite(f'../datasets/aligned/96x112/{mset}/{file}', face_96x112)

                                    face_112x112 = alignment(cv_img, dst, 112, 112)
                                    cv2.imwrite(f'../datasets/aligned/112x112/{mset}/{file}', face_112x112)

                                    face_150x150 = alignment(cv_img, dst, 150, 150)
                                    cv2.imwrite(f'../datasets/aligned/150x150/{mset}/{file}', face_150x150)

                                    face_160x160 = alignment(cv_img, dst, 160, 160)
                                    cv2.imwrite(f'../datasets/aligned/160x160/{mset}/{file}', face_160x160)

                                    face_224x224 = alignment(cv_img, dst, 224, 224)
                                    cv2.imwrite(f'../datasets/aligned/224x224/{mset}/{file}', face_224x224)

                                check = True
                                break

                    else:
                        for ix, point in enumerate(landmarks):
                            dst = np.float32(point)
                            cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            file = "{}.{}.jpg".format(file.split(".jpg")[0], ix + 1)

                            face_96x112 = alignment(cv_img, dst, 96, 112)
                            cv2.imwrite(f'../datasets/aligned/96x112/{mset}/{file}', face_96x112)

                            face_112x112 = alignment(cv_img, dst, 112, 112)
                            cv2.imwrite(f'../datasets/aligned/112x112/{mset}/{file}', face_112x112)

                            face_150x150 = alignment(cv_img, dst, 150, 150)
                            cv2.imwrite(f'../datasets/aligned/150x150/{mset}/{file}', face_150x150)

                            face_160x160 = alignment(cv_img, dst, 160, 160)
                            cv2.imwrite(f'../datasets/aligned/160x160/{mset}/{file}', face_160x160)

                            face_224x224 = alignment(cv_img, dst, 224, 224)
                            cv2.imwrite(f'../datasets/aligned/224x224/{mset}/{file}', face_224x224)

                        check = True

                    if check == False:
                        count += 1
                        print(img_path + '\t' + 'corrupted')
                        unknown_file.write(file + '\n')

                        copyfile(img_path, f'../datasets/unknown/{file}')

                    total += 1
                except:
                    not_aligned_file.write(file + '\n')
                    copyfile(img_path, f'../datasets/not_aligned/{file}')

        unknown_file.close()
        not_aligned_file.close()

    for mset in ['train', 'test']:
        if os.path.exists(f'../datasets/aligned/224x224'):
            for file in tqdm(os.listdir(f'../datasets/aligned/224x224/{mset}')):
                img = cv2.imread(f'../datasets/aligned/224x224/{mset}/{file}')
                gender = detect_gender(img)
                new_file = "{}.{}.jpg".format(file.split(".jpg")[0], gender)
                os.rename(f'../datasets/aligned/224x224/{mset}/{file}',
                          f'../datasets/aligned/224x224/{mset}/{new_file}')
                os.rename(f'../datasets/aligned/160x160/{mset}/{file}',
                          f'../datasets/aligned/160x160/{mset}/{new_file}')
                os.rename(f'../datasets/aligned/150x150/{mset}/{file}',
                          f'../datasets/aligned/150x150/{mset}/{new_file}')
                os.rename(f'../datasets/aligned/112x112/{mset}/{file}',
                          f'../datasets/aligned/112x112/{mset}/{new_file}')
                os.rename(f'../datasets/aligned/96x112/{mset}/{file}',
                          f'../datasets/aligned/96x112/{mset}/{new_file}')

    print("[INFO] {}, {}".format(count, total))


if __name__ == "__main__":
    main()
