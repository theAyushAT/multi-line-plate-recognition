import math
import time
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.meters import AverageMeter
import argparse
from torchvision import transforms as T
from model import PlateNet
import tensorflow.keras.backend as K

chars = [ u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H",u"I", u"J", u"K", u"L", u"M", u"N",u"O", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z"]

def preprocess_image(image):
    image = T.ToTensor()(image)
    image = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(image)
    return torch.unsqueeze(image, dim=0)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_images",
        type=str,
        required=True,
        default="./",
        help="Path where all the test images are located or you can give path to video, it will break into each frame and write as a video",
    )

    parser.add_argument(
        "--path_to_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to save checkpoints and wandb, final output path will be this path + wandbexperiment name so the output_dir should be root directory",
    )

    model = PlateNet(batch_size=opt.train_batch, n_class=36)
    model.load_state_dict(torch.load(args.path_to_weights)["state_dict"])
    model.eval()

    right_num = 0   
    for image_path in tqdm(os.listdir(args.path_to_images)):
        label = image_path[:-4]
        image = cv2.imread(os.path.join(args.path_to_images, image_path))
        image = preprocess_image(image, cfg)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            log_probs = model(image,1)
            y_pred = nn.functional.softmax(log_probs,dim =0)
            shape = y_pred[:, :, :].shape
            out = K.get_value(K.ctc_decode(y_pred[:,:,:],input_length=np.ones(shape[0])*shape[1])[0][0])[:,
                :11]

            eco = len(chars)+1
            str_label=''.join([str(x) for x in label if x!=eco])
            str_out = ''.join([str(x) for x in out if x!=eco])
            if str_label == str_out:
                right_num+=1
    
    acc = (right_num / len(os.listdir(args.path_to_images))) * 100
    print("test acc is :{}%".format(str(acc)))
