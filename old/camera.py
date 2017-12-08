import numpy as np

class Camera(object):
    def __init__(self, env, width=200, height=200, name="my_camera"):
        self.frame = 0
        self.width = width
        self.height = height
        self.x, self.y, self.z = 1, 1, 1
        self.tx, self.ty, self.tz = 0, 0, 0
        self.cam = env.unwrapped.scene.cpp_world.new_camera_free_float(self.height, self.width, name)

    def observation(self, depth=False, label=False, verbose=False):
        ''' function that extracts pixel values for rgb, depth and labels
        :param depth        boolean, get depth pixels
        :param label        boolean, get label pixels
        :param verbose      boolean, print time for this process

        :output rgb         rgb_array, np.array.dtype('uint8')
        :output depth       rgb_array, np.array.dtype('float32')
        :output label       rgb_array, np.array.dtype('uint8')
        '''
        self.frame += 1

        rgb, depth, depth_mask, labeling, labeling_mask = self.cam.render(depth, label, verbose)
        rgb = np.fromstring(rgb, dtype=np.uint8).reshape((self.width, self.height, 3) )
        if depth:
            depth = np.fromstring(depth, dtype=np.float32).reshape((self.width // 2 , self.height // 2, 1) )
        if labeling:
            label = np.fromstring(labeling, dtype=np.uint8).reshape((self.width // 2, self.height // 2, 1) )
        return rgb, depth, labeling

    def look_from_random_pos(self):
        x, y, z = np.random.rand(3)*2 + 1
        self.cam.move_and_look_at(x,y,z, 0,0,0)

    def look_at(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.cam.move_and_look_at(self.x, self.y, self.z,
                                  self.tx, self.ty, self.tz)

    def change_position(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.cam.move_and_look_at(x, y, z,
                                  self.tx, self.ty, self.tz)

    def circle_target(self, r=1, turn=500):
        ''' Changes position of camera in a circle with radius r.
            z-axis is fixed
        :param r        scalar, radius of circle in xy-plane
        :param turn     scalar, view angles/revolution.
        '''
        turn = 2 * 3.14 / turn
        theta = self.frame*turn*3.14
        x, y = r * np.cos(theta), r * np.sin(theta)
        self.change_position(x,y,self.z)

def test():
    import gym
    import roboschool
    import cv2

    from OpenGL import GLU # fix for opengl issues on desktop  / nvidia
    # env.set_initial_orientation(task=1, yaw_center=0, yaw_random_spread=1)

    steps = 100
    turn = 3.14/500

    env = gym.make('RoboschoolReacher-v1')
    s = env.reset()
    env.unwrapped.initial_z = 8
    print(env.unwrapped.initial_z)

    cam = Camera(env)

    for j in range(5):
        env.reset()
        for i in range(steps):
            print('step:', i)
            # env.render()
            # obs, depth, label = cam.observation()
            # cv2.imshow('depth', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            env.render()
            # Press "q" to quit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
            cam.circle_target()
            a = env.action_space.sample()
            s, r, d, i = env.step(a)

if __name__=="__main__":
    test()
