import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils.utils import *

# YOLO added
from model import yolov3
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize


import tellopy
import numpy
import av
import cv2
from pynput import keyboard
from tracker import Tracker
from djitellopy import Tello
from PIL import Image



#forces tensorflow to run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""

parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")

args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

#upup = True 

def main(_):
   
  tellotrack = TelloCV()



  with tf.Graph().as_default():
    width, height = args.new_size[0], args.new_size[1]

    #cam = cv2.VideoCapture(0)
    print(tellotrack.takeoff_time)
    #if time.time() - tellotrack.takeoff_time > 6 :    
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32,[1, width, height, 3], name='input_data')

        yolo_model = yolov3(args.num_class, args.anchors)

        with tf.variable_scope('yolov3'):
          pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.7, nms_thresh=0.7)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        tellotrack.drone.takeoff()

        find_pad = False
        landing = False
                
        while True:
            img_ori = tellotrack.process_frame()

            print("IMG SHAPE : ", img_ori.shape)
            img = cv2.resize(img_ori, (width, height))
            print("RESIZE IMg SHAPE : ", img.shape)
            img=np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.
            print("EXPAND IMG SHAPE : ", img.shape)
            print(img.shape)
            start = time.time()


            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            print(boxes_)
            end = time.time()
            boxes_[:, [0, 2]] *= (img_ori.shape[1]/float(width))
            boxes_[:, [1, 3]] *= (img_ori.shape[0]/float(height))

            
            for i in range(len(boxes_)):
              x0, y0, x1, y1 = boxes_[i]
              #plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
              if labels_[i] == 0: #56 chair 11 stopsign 0 person 74 clock
                print("-------------------------------------FIND-----")
                plot_one_box(img_ori, [x0,y0,x1,y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color = color_table[labels_[i]])

                '''
                move tello to center of object
                move_done = tellotrack.track_mid((x1+x0) / 2, (y1+y0) / 2)
                if move_done:
                    tellotrack.drone.land()

                '''

                # move left and find mission pad, landing.
                if int(x1-x0) > 200 and find_pad == False:
                    #tellotrack.move_left()
                    #time.sleep(3)
                    #tellotrack.move_left()
                    #time.sleep(3)
                    tellotrack.drone.send_control_command("mon")
                    tellotrack.drone.send_control_command("mdirection 0")
                    find_pad = True
                else:
                    tellotrack.go()
                    time.sleep(3)

            if find_pad and landing is False:
                response = tellotrack.drone.get_current_state_all()
                print(response)
                time.sleep(5)
                response = tellotrack.drone.send_control_command("go 0 0 70 30 m2")
                time.sleep(5)
                if response is False:
                    tellotrack.go_slow()
                else:
                    landing = True

            if landing:
                tellotrack.landing()
                time.sleep(5)
                tellotrack.take_off()
                landing = False

            cv2.imshow('YOLO', img_ori)

            k = cv2.waitKey(1)         
            if k == 1048603 or k == 27: 
              break  # esc to quit
            if k == 1048688:
              cv2.waitKey(0) # 'p' to pause
            if k == ord('h') :
                tellotrack.tracking = True # 'h' to use tracking
                tellotrack.track_cmd = ""

            print("Time: " + str(end - start))
            del img
            del img_ori
            #del toShow
          
        #cam.release()        





class TelloCV(object):
    """
    TelloTracker builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self):
        self.prev_flight_data = None
        self.record = False
        self.tracking = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.speed = 50
        self.go_speed = 90
        self.drone = Tello()
        self.init_drone()
        self.init_controls()
        
        self.battery = self.drone.get_battery()
        self.frame_read = self.drone.get_frame_read()
        self.forward_time = 0
        self.forward_flag = True
        self.takeoff_time = 0
        self.command_time = 0
        self.command_flag = False

        # trackingimport libh264decoder a color
        #green_lower = (30, 50, 50)
        #green_upper = (80, 255, 255)
        #red_lower = (0, 50, 50)
        #red_upper = (20, 255, 255)
        blue_lower = np.array([0,0,0])
        upper_blue = np.array([255,255,180])
        bh_lower = (180,30,100)
        bh_upper=(275,50,100)
        self.track_cmd = ""
        self.tracker = Tracker(960,
                               720,
                               blue_lower, upper_blue)
        self.speed_list = [5, 10, 15, 20, 25]
    def init_drone(self):
        """Connect, uneable streaming and subscribe to events"""
        # self.drone.log.set_level(2)
        self.drone.connect()
        self.drone.streamon()


    def on_press(self, keyname):
        """handler for keyboard listener"""
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.quit()
                exit(0)
            if keyname in self.controls:
                key_handler = self.controls[keyname]
                if isinstance(key_handler, str):
                    getattr(self.drone, key_handler)(self.speed)
                else:
                    key_handler(self.speed)
        except AttributeError:
            print('special key {0} pressed'.format(keyname))

    def on_release(self, keyname):
        """Reset on key up from keyboard listener"""
        self.keydown = False
        keyname = str(keyname).strip('\'')
        print('-' + keyname)
        if keyname in self.controls:
            key_handler = self.controls[keyname]
            if isinstance(key_handler, str):
                getattr(self.drone, key_handler)(0)
            else:
                key_handler(0)

   # def takeoff_f(self):
    #    print( "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
     #   while self.takeoff_time < 7 :
           # print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
            #getattr(self.drone, "takeoff")(self.takeoff_speed)
           # lambda speed: self.drone.takeoff(self.takeoff_speed)

            #self.takeoff_time = self.takeoff_time + 1

    def init_controls(self):
        
        """Define keys and add listener"""
        self.controls = {
            'w': lambda speed: self.drone.move_forward(speed),
            's': lambda speed: self.drone.move_back(speed),
            'a': lambda speed: self.drone.move_left(speed),
            'd': lambda speed: self.drone.move_right(speed),
            'Key.space': 'up',
            'Key.shift': 'down',
            'Key.shift_r': 'down',
            'q': 'counter_clockwise',
            'e': 'clockwise',
            'i': lambda speed: self.drone.flip_forward(),
            'k': lambda speed: self.drone.flip_back(),
            'j': lambda speed: self.drone.flip_left(),
            'l': lambda speed: self.drone.flip_right(),
            # arrow keys for fast turns and altitude adjustments
            'Key.left': lambda speed: self.drone.rotate_counter_clockwise(speed),
            'Key.right': lambda speed: self.drone.rotate_clockwise(speed),
            'Key.up': lambda speed: self.drone.move_up(speed),
            'Key.down': lambda speed: self.drone.move_down(speed),
            'Key.tab': lambda speed: self.drone.takeoff(),
            #'Key.tab': self.drone.takeoff(60),
            'Key.backspace': lambda speed: self.drone.land(),
            'p': lambda speed: self.palm_land(speed),
            't': lambda speed: self.toggle_tracking(speed),
            'r': lambda speed: self.toggle_recording(speed),
            'z': lambda speed: self.toggle_zoom(speed),
            'Key.enter': lambda speed: self.take_picture(speed)
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()
        # self.key_listener.join()

   

    def process_frame(self):
        """convert frame to cv2 image and show"""
        print("TRACKING START")
        frame = self.frame_read.frame
        #self.drone.move_up(self.speed)
        #image = self.write_hud(image)
        #if self.record:
        #    self.record_vid(frame)
        return frame
               
    def move_up(self):
        self.drone.move_up(self.speed)
    
    def take_off(self):
        self.drone.takeoff()
    
    def go(self):
        self.drone.move_forward(self.go_speed)
    def move_left(self):
        self.drone.move_left(self.go_speed) # speed 테스트해서 조절하기
    def go_window9(self):
        self.drone.move_forward()
    def rotate_right(self):
        self.drone.rotate_clockwise()
    def rotate_left(self):
        self.drone.rotate_counter_clockwise()
    def landing(self):
        self.drone.land()

    def tracking_f(self,image):
        #if upup : 
        #    self.drone.move_up(self.speed)

        xoff, yoff = self.tracker.track(image)
        #print(xoff, yoff)
        image = self.tracker.draw_arrows(image)

        distance = 30
        cmd = ""
        #self.drone.takeoff()
                
        if time.time() - self.foward_time > 3 and self.forward_flag:
            self.drone.move_forward(self.takeoff_speed)
            self.forward_time = time.time()
            self.forward_flag = False
            self.command_flag = True
        elif time.time() - self.command_time > 3 and self.command_flag:
        #if (time.time() - self.forward_time) > 5 :
            #self.drone.move_forward(self.speed)
            if xoff < (-distance):
                print("11111111111111111")
                self.drone.move_left(self.speed)
            elif xoff > distance :
                print("2222222222222222")
                self.drone.move_right(self.speed)
            self.command_time = time.time()
            self.command_flag = False
            self.forward_flag = True
            #self.forward_time = time.time()

            #    print("333333333333333")
             #   self.drone.move_down(self.speed)
              #  self.drone.forward(self.speed)
              #  cmd = "move_down"
            #elif yoff > distance:
             #   print("444444444444444")
              #  self.drone.move_up(self.speed)
                
               # cmd = "move_up"
            #else:
                #if self.track_cmd is not "":
                 #   print(self.track_cmd)
                #    getattr(self.drone, self.track_cmd)(0)
                 #   self.track_cmd = ""
                
                #if time.time() - self.forward_time > 5:
                 #   print("move foward")
                  #  self.forward_time = time.time()
                   # getattr(self.drone, "move_forward")(self.speed)


        #if cmd is not self.track_cmd:
         #   print("||||||||||||||||||||||")
          #  if cmd is not "" :
           #     print("track command:", cmd)
            #    getattr(self.drone, cmd)(self.speed)
             #   self.track_cmd = cmd
        


    def write_hud(self, frame):
        """Draw drone info, tracking and record on frame"""
        stats = self.prev_flight_data.split('|')
        stats.append("Tracking:" + str(self.tracking))
        if self.drone.zoom:
            stats.append("VID")
        else:
            stats.append("PIC")
        if self.record:
            diff = int(time.time() - self.start_time)
            mins, secs = divmod(diff, 60)
            stats.append("REC {:02d}:{:02d}".format(mins, secs))

        for idx, stat in enumerate(stats):
            text = stat.lstrip()
            cv2.putText(frame, text, (0, 30 + (idx * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), lineType=30)
        return frame

    def toggle_recording(self, speed):
        """Handle recording keypress, creates output stream and file"""
        if speed == 0:
            return
        self.record = not self.record

        if self.record:
            datename = [os.getenv('HOME'), datetime.datetime.now().strftime(self.date_fmt)]
            self.out_name = '{}/Pictures/tello-{}.mp4'.format(*datename)
            print("Outputting video to:", self.out_name)
            self.out_file = av.open(self.out_name, 'w')
            self.start_time = time.time()
            self.out_stream = self.out_file.add_stream(
                'mpeg4', self.vid_stream.rate)
            self.out_stream.pix_fmt = 'yuv420p'
            self.out_stream.width = self.vid_stream.width
            self.out_stream.height = self.vid_stream.height

        if not self.record:
            print("Video saved to ", self.out_name)
            self.out_file.close()
            self.out_stream = None

    def record_vid(self, frame):
        """
        convert frames to packets and write to file
        """
        new_frame = av.VideoFrame(
            width=frame.width, height=frame.height, format=frame.format.name)
        for i in range(len(frame.planes)):
            new_frame.planes[i].update(frame.planes[i])
        pkt = None
        try:
            pkt = self.out_stream.encode(new_frame)
        except IOError as err:
            print("encoding failed: {0}".format(err))
        if pkt is not None:
            try:
                self.out_file.mux(pkt)
            except IOError:
                print('mux failed: ' + str(pkt))

    def take_picture(self, speed):
        """Tell drone to take picture, image sent to file handler"""
        if speed == 0:
            return
        self.drone.take_picture()

    def palm_land(self, speed):
        """Tell drone to land"""
        if speed == 0:
            return
        self.drone.palm_land()

    def toggle_tracking(self, speed):
        """ Handle tracking keypress"""
        if speed == 0:  # handle key up event
            return
        self.tracking = not self.tracking
        print("tracking:", self.tracking)
        return

    def toggle_zoom(self, speed):
        """
        In "video" mode the self.drone sends 1280x720 frames.
        In "photo" mode it sends 2592x1936 (952x720) frames.
        The video will always be centered in the window.
        In photo mode, if we keep the window at 1280x720 that gives us ~160px on
        each side for status information, which is ample.
        Video mode is harder because then we need to abandon the 16:9 display size
        if we want to put the HUD next to the video.
        """
        if speed == 0:
            return
        self.drone.set_video_mode(not self.drone.zoom)

    def flight_data_handler(self, event, sender, data):
        """Listener to flight data from the drone."""
        text = str(data)
        if self.prev_flight_data != text:
            self.prev_flight_data = text

    def handle_flight_received(self, event, sender, data):
        """Create a file in ~/Pictures/ to receive image from the drone"""
        path = '%s/Pictures/tello-%s.jpeg' % (
            os.getenv('HOME'),
            datetime.datetime.now().strftime(self.date_fmt))
        with open(path, 'wb') as out_file:
            out_file.write(data)
        print('Saved photo to %s' % path)
       
    def enable_mission_pads1(self):
        self.drone.mon()
    def disable_mission_pads1(self):
        self.drone.moff()
    def go_xyz_speed_mid1(self,x,y,z, speed, mid):
        self.drone.go_x_y_z_speed_mid(x,y,z,speed,mid)

    #  if function return True, set drone center to object's center
    def track_mid(self, x, y):
        midx, midy = 480, 360
        distance_x = abs(midx - x)
        distance_y = abs(midy - y)
        print(x, y, distance_x, distance_y)
        speed_index_x = int(distance_x % 10) - 1
        if speed_index_x > 4:
            speed_index_x = 4
        speed_index_y = int(distance_y % 10) - 1
        if speed_index_y > 4:
            speed_index_y =4
        move_done = True
        if y > midy:
            self.drone.move_down(self.speed_list[speed_index_y])
            move_done = False
        elif y < midy:
            self.drone.move_up(self.speed_list[speed_index_y])
            move_done = False
        elif x > midx:
            self.drone.move_left(self.speed_list[speed_index_x])
            move_done = False
        elif x < midx:
            self.drone.move_right(self.speed_list[speed_index_x])
            move_done = False
        return move_done

    def go_slow(self):
        self.drone.move_forward(30)


if __name__ == '__main__':
    tf.app.run()
