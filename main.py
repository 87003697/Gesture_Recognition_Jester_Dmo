import  cv2
import os
from lib.model_ResNet3D import Resnet3DBuilder
from lib.data_loader import frame_queue
from PIL import Image as pil_image
import csv
import numpy as np

def main():
    # activate camera on laptop
    cap = cv2.VideoCapture(0)

    #  Importing Hdf5 Model Of ResNet3D-101 ##############################
    # some of key values show below, latter save them in config.cfg

    code_root = '/home/zhiyuanma/Desktop/Academic_Action_Recognition'
    model_name = 'model.best.hdf5'
    width = 640
    height = 480
    nb_frames = 16
    target_size = (64,96)
    nb_classes = 27

    # inp_shape = (None, None, None, 3) = frames length width channels
    inp_shape = (nb_frames,) + target_size + (3,)

    #loading model ResNet3D 101
    net = Resnet3DBuilder.build_resnet_101(inp_shape,nb_classes)
    net.load_weights(model_name)


    # Setting  Values Of Cap ##############################################
    cap.set(3,960)
    cap.set(4,640)
    cap.set(11,0) # Brightness
    cap.set(12,100) # Contrast

    # Wring In Label #####################################################
    with open('jester-v1-labels.csv')as f:
        f_csv = csv.reader(f)
        label_list = []
        for row in f_csv:
            label_list.append(row)
        label_list = tuple(label_list)

    # Initialize Class ###################################################
    Queue = frame_queue(nb_frames, target_size)

    while(cap.isOpened()):
        on,frame = cap.read()
        if on == True:

            # Calibrating Channels ########################################
            b, g, r = cv2.split(frame)
            frame_cali = cv2.merge([r, g, b])

            # Build Queues As Input ########################################

            batch_x = Queue.img_inQueue(frame)
            # predict results ######################################################
            res = net.predict(batch_x)
            res = list(res[0])
            index = res.index(max(res))
            print(label_list[index])


            cv2.imshow('camera0', frame)

            # exit when pressing key Q
            if(cv2.waitKey(1)&0xFF) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# Copy some codes of data_loader.py for the moment

if __name__ == '__main__':
    main()







