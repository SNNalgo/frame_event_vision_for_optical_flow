import tonic
import numpy as np
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time


#recording = "slider_depth"
#start_event_frame = 8
#end_event_frame = 640

recording = "shapes_6dof"
start_event_frame = 8
end_event_frame = 1600

#recording = "dynamic_translation"
#start_event_frame = 8
#end_event_frame = 1600

#recording = "dynamic_6dof"
#start_event_frame = 1600
#end_event_frame = 3200

dataset = tonic.datasets.DAVISDATA(save_to="data", recording=recording)

data, targets = dataset[0]
events, imu, images = data

print(events)
print("event x : ", events["x"], "event x shape : ", events["x"].shape)
print("event y : ", events["y"], "event y shape : ", events["y"].shape)
print("event t : ", events["t"], "event t shape : ", events["t"].shape)
print("event p : ", events["p"], "event p shape : ", events["p"].shape)
print("key of imu: ", imu.keys())
print("key of images: ", images.keys())
print("key of targets: ", targets.keys())
print("imu rotQ shape : ", imu["rotQ"].shape)
print("imu angV shape : ", imu["angV"].shape)
print("imu acc shape : ", imu["acc"].shape)
print("imu mag shape : ", imu["mag"].shape)
print("imu rotQ max : ", np.max(imu["rotQ"]), "imu rotQ min : ", np.min(imu["rotQ"]))
print("imu angV max : ", np.max(imu["angV"]), "imu angV min : ", np.min(imu["angV"]))
print("imu acc max : ", np.max(imu["acc"]), "imu acc min : ", np.min(imu["acc"]))
print("imu mag max : ", np.max(imu["mag"]), "imu mag min : ", np.min(imu["mag"]))
print("images frames shape : ", images["frames"].shape)
print("targets rotation shape : ", targets["rotation"].shape)
print("targets point shape : ", targets["point"].shape)

print('total length of recording (s): ', 1e-6*images["ts"][-1])

mean_diff = np.diff(list(zip(images["ts"], images["ts"][1:]))).mean()
print("Average difference in image timestamps in microseconds: ", mean_diff)

im_frame_latency = float("{:.2f}".format(1e-3*(mean_diff))) #in ms
event_frame_latency = float("{:.2f}".format(1e-3*(mean_diff/8))) #in ms

sensor_size = tonic.datasets.DAVISDATA.sensor_size
frame_transform = tonic.transforms.Compose([
    tonic.transforms.Denoise(filter_time=event_frame_latency*1e3),
    tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=event_frame_latency*1e3)# in us
    ])

image_center_crop = torchvision.transforms.Compose(
    [torch.tensor, torchvision.transforms.CenterCrop((150, 150))]
)

def data_transform(data):
    # first we have to unpack our data
    events, imu, images = data
    # we bin events to event frames
    event_frames = frame_transform(events)
    image_frames = images["frames"]
    # then we can apply frame transforms to both event frames and images at the same time
    #event_frames = image_center_crop(event_frames)
    #image_frames = image_center_crop(image_frames)
    return event_frames, imu, image_frames

dataset = tonic.datasets.DAVISDATA(
    save_to="./data", recording=recording, transform=data_transform
)

data, targets = dataset[0]
event_frames, imu, image_frames = data

print('event_frames shape: ', event_frames.shape)
print('image_frames shape: ', image_frames.shape)

fig, [(ax1, ax2, ax3),(ax4, ax5, ax6)] = plt.subplots(2, 3)
event_frame = event_frames[80]

im1 = ax1.imshow(event_frame[0] - event_frame[1])
ax1.set_title("event frame, frame latency: " + str(event_frame_latency)+ 'ms')

im4 = ax4.imshow(image_frames[10], cmap=mpl.cm.gray)
ax4.set_title("optical frame, frame latency: " + str(im_frame_latency)+ 'ms');
#print(image_frames[10].dtype)

im2 = ax2.imshow(event_frame[0] - event_frame[1])
ax2.set_title("event flow y")

im5 = ax5.imshow(event_frame[0] - event_frame[1])
ax5.set_title("event flow x")

im3 = ax3.imshow(event_frame[0] - event_frame[1])
ax3.set_title("optical flow y")

im6 = ax6.imshow(event_frame[0] - event_frame[1])
ax6.set_title("optical flow x")

decay_fac = 0.9
eps = 1e-6
consistency_thresh = 1.0
flow_thresh = 1.0
#confidence_map_h = 0
#confidence_map_v = 0
#confidence_decay = 0.8
#confidence_thresh = 1.0

event_frame_pot = 0*(event_frame[0] - event_frame[1])
event_flow_h_prev = event_frame[0] - event_frame[1]
event_flow_v_prev = event_frame[0] - event_frame[1]
event_flow_h = 0*event_flow_h_prev
event_flow_v = 0*event_flow_v_prev
event_proc_time = 0

plt.ion()

for i in range(start_event_frame, end_event_frame):
    #Optical frame processing
    if i%8 == 0:
        opt_proc_start = time.time_ns()
        image_frame = image_frames[i//8]
        flow = cv2.calcOpticalFlowFarneback(image_frames[(i//8)-1], image_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        opt_proc_end = time.time_ns()
        opt_proc_time = float("{:.2f}".format(1e-6*(opt_proc_end - opt_proc_start)))
        
        im4.set_data(image_frame)
        im3.set_data(flow[...,1]) #optical flow h
        im6.set_data(flow[...,0]) #optical flow v
        flow[...,0] = ndimage.uniform_filter(flow[...,0], size=5)
        flow[...,1] = ndimage.uniform_filter(flow[...,1], size=5)
        ax2.set_title("valid event flow y, proc time: " + str(event_proc_time) + 'ms')
        ax5.set_title("valid event flow x, proc time: " + str(event_proc_time) + 'ms')
        ax3.set_title("optical flow y, proc time: " + str(opt_proc_time) + 'ms')
        ax6.set_title("optical flow x, proc time: " + str(opt_proc_time) + 'ms')
    
    event_frame = event_frames[i]
    event_proc_start = time.time_ns()
    
    #Leaky integrator of events
    event_frame_pot = event_frame_pot*decay_fac + (event_frame[1] + event_frame[0])
    
    #Get raw horizontal and vertical flows using Sobel
    event_flow_h = ndimage.sobel(event_frame_pot, 0)  # horizontal gradient
    event_flow_v = ndimage.sobel(event_frame_pot, 1)  # vertical gradient
    
    event_flow_dist = np.sqrt((event_flow_h-event_flow_h_prev)**2 + (event_flow_v-event_flow_v_prev)**2)
    event_flow_mask = np.float32(event_flow_dist<consistency_thresh)
    event_flow_h_prev = event_flow_h
    event_flow_v_prev = event_flow_v
    
    event_flow_h_masked = event_flow_mask*event_flow_h
    event_flow_v_masked = event_flow_mask*event_flow_v
    
    #event_opt_sign_mask_v = np.float32((flow[...,0]*event_flow_v_masked)>0)
    #event_opt_dist = np.sqrt((event_flow_h_masked-flow[...,1])**2 + (event_flow_v_masked-flow[...,0])**2)
    #event_opt_val_mask_h = np.float32(event_opt_dist<(flow_thresh*np.sqrt((flow[...,1])**2 + (flow[...,0])**2)))
    #event_opt_val_mask_v = event_opt_val_mask_h
    event_opt_val_mask_h = np.float32(np.abs(flow[...,1]-event_flow_h_masked)<(flow_thresh*np.abs(flow[...,1])))
    event_opt_val_mask_v = np.float32(np.abs(flow[...,0]-event_flow_v_masked)<(flow_thresh*np.abs(flow[...,0])))
    
    event_flow_h_masked = (event_opt_val_mask_h)*event_flow_h_masked
    event_flow_v_masked = (event_opt_val_mask_v)*event_flow_v_masked
    
    event_proc_end = time.time_ns()
    event_proc_time = float("{:.2f}".format(1e-6*(event_proc_end - event_proc_start)))
    
    im1.set_data(event_frame_pot)
    im2.set_data(event_flow_h_masked)
    im5.set_data(event_flow_v_masked)
    plt.pause(0.001)

plt.ioff()
plt.show()