import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *
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
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')

args = parser.parse_args()

#upup = True 

def main(_):
   
  tellotrack = TelloCV()
  
  with tf.Graph().as_default():
    height = args.height
    width = args.width
    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

    with tf.variable_scope("model") as scope:
      model = pydnet(placeholders)

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    #cam = cv2.VideoCapture(0)
    print(tellotrack.takeoff_time)
    #if time.time() - tellotrack.takeoff_time > 6 :    
    tellotrack.take_off()
    time.sleep(7)
       # tellotrack.takeoff_time = time.time()

    tellotrack.move_up()
    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        #tellotrack.move_up()
       # tellotrack.takeoff_f()
        while True:
           # if tellotrack.upup :
            #    tellotrack.move_up()
            img = tellotrack.process_frame()
            #if tellotrack.upup :
            #    tellotrack.move_up()

            print("IMG SHAPE : ", img.shape)

            img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
            print("RESIZE IMg SHAPE : ", img.shape)
            img = np.expand_dims(img, 0)
            print("EXPAND IMG SHAPE : ", img.shape)
            print(img.shape)
            start = time.time()
            disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
            end = time.time()
            #disp_img = Image.fromarray(disp[0],"RGB")
            rows, cols =disp.shape[1:3]
            disp_color = (applyColorMap(disp[0,:,:,0]*5, 'jet')*255.).astype(np.uint8)
            tellotrack.tracking_f(disp_color)
            toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)
            #toShow = cv2.resize(toShow, (int(width/2), height))
            #cv2.imwrite("sample_{}.jpg".format(time.time()), toShow)
            #cv2.imshow('disp',disp_img)
             
            cv2.imshow('pydnet', toShow)
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
            del disp
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
        self.speed = 30
        self.takeoff_speed = 90
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
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #self.drone.move_up(self.speed)
        #image = self.write_hud(image)
        #if self.record:
        #    self.record_vid(frame)
        return image
    def take_off(self):
        self.drone.takeoff()
               
    def move_up(self):
        self.drone.move_up(self.speed)
        

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




if __name__ == '__main__':
    tf.app.run()
