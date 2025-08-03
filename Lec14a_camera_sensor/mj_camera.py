import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import cv2

xml_path = 'block_with_camera.xml' #xml file (assumes this is in the same folder as this file)
simend = 20 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)
print_model = 0 #set to 1 to print the model info in the file model.txt in the current location

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

#print the model
if (print_model==1):
    model_name = 'model.txt'
    model_path = os.path.join(dirname + "/" + model_name)
    mujoco.mj_printModel(model,model_path)


# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        #mj.mj_step(model, data)
        qz = 0.25*np.sin(data.time);
        q0 = np.sqrt(1-qz*qz)
        quat = np.array([q0,0,0,qz])
        data.time += model.opt.timestep
        data.qpos[3:] = quat.copy()
        mj.mj_forward(model,data)
    

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    #Code taken from https://github.com/dtorre38/mujoco_opencv
    # ******** inset view (code start) *********
    #Settings for inset view
    camera_name = 'robot_camera'
    width = 0.4*640
    height = 0.4*480

    loc_x=int(viewport_width - width)
    loc_y=int(viewport_height - height)
    height = int(height)
    width = int(width)

    # 1. Create a rectangular viewport in the upper right corner for example.
    offscreen_viewport = mj.MjrRect(loc_x, loc_y, width, height)

    # 2. Set the camera to the specified view
    camera_id = model.camera(camera_name).id
    offscreen_cam = mj.MjvCamera()
    offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    offscreen_cam.fixedcamid = camera_id

    #3. Update scene for the off-screen camera
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # 4. Render the scene in the offscreen buffer with mjr_render.
    mj.mjr_render(offscreen_viewport, scene, context)

    # ******** inset view (code end) *********

    # ********* Open CV code starts ********
    #This example shows how to transform the image to grayscale.
    #This code was generated by ChatGPT

    # Capture the pixels from the robot_camera render
    rgb_pixels = np.zeros((height, width, 3), dtype=np.uint8)
    depth_pixels = np.zeros((height, width), dtype=np.float32)
    mj.mjr_readPixels(rgb_pixels, depth_pixels, offscreen_viewport, context)

    # Convert the RGB image to grayscale (black-and-white)
    gray_pixels = np.dot(rgb_pixels[..., :3], [0.2989, 0.5870, 0.1140])  # Using standard luminosity formula

    # Convert back to 3 channels for rendering
    bw_pixels = np.stack([gray_pixels] * 3, axis=-1).astype(np.uint8)

    # Flip the grayscale image vertically to match OpenCV's image coordinate system
    bw_pixels_flipped = cv2.flip(bw_pixels, 0)  # Flip vertically (0 means flip vertically)

    # Convert bw_pixels to an image format OpenCV can use (BGR format)
    bw_image_bgr = cv2.cvtColor(bw_pixels_flipped, cv2.COLOR_RGB2BGR)

    # Show the grayscale image in the inset location (in OpenCV window)
    cv2.imshow("Black and White Image", bw_image_bgr)
    cv2.waitKey(1)  # Update display with a short delay

    # *********** Open CV code ends ********


    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
