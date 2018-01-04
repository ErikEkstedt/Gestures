# Environment

In my custom classes i load the xml file for the robot only once. Only if no
scene has been initialized. For me this makes the reset() work as expected.

```python
def _reset(self):
	if self.scene is None:
		''' First reset '''
		self.scene = self.initialize_scene()
		self.load_xml_get_robot()

	self.get_join_dicts()
	self.robot_specific_reset()

	# Important Resets
	self.done = False
	self.frame = 0

	for r in self.mjcf:
		r.query_position()

	self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
	s = self.calc_state()
	self.potential = self.calc_potential()
	self.camera_adjust()
	rgb = self.get_rgb()
	return s, rgb
```

I do not have great knowledge about everything in the bullet/gym code but it
seems superfluous to load the xml file each reset. At least in the singleplayer
scenario, I have not tried multiplayer.

## reacher.py

### class Base(MyGymEnv)
Base class inherets from [MyGymEnv](environments/my_gym_env.py)

### class ReacherPlaneNoTarget(Base) 
No target included in render or state. A bare bone baseline environment which
returns a state with key_robot_pos and joint_speeds

| Action size | State size | State Information |
|:-----------:|:--------:|:-------:|
| 2						| 6			| elbow_XY, hand_XY, joint speed |


### class ReacherCommon()

### class ReacherPlane(ReacherCommon, Base)
* No z-axis info (constrained in plane)
* calc_state  -> robot_key_points, joint_speeds
* calc_reward -> None

| Action size | State size | State Information |
|:-----------:|:--------:|:-------:|
| 2						| 14			| elbow_XY, hand_XY, target1_XY, target2_XY, joint speed |


### class Reacher3D(ReacherCommon, Base)

| Action size | State size | State Information |
|:-----------:|:--------:|:-------:|
| 3						| 21			| elbow XYZ, hand_XYZ, joint speed |

## humanoid.py
* class MyGymEnv(gym.Env)
* class Base(MyGymEnv)
* class TargetHumanoid(Base)
* class Humanoid(Base)

## my_gym_env.py
MyGymEnv is a class which is the wrapper for OpenAI gym. It contains important
functions such as:
* _seed
* _reset
* _render
* _step

        
## SubProcEnv.py

* class CloudpickleWrapper(object):
* class SubprocVecEnv(VecEnv):
* class SubprocVecEnv_RGB(VecEnv):
* def worker_RGB(remote, parent_remote, env_fn_wrapper):
* def worker(remote, parent_remote, env_fn_wrapper):

## utils.py
* def rgb_render(obs, title='obs')
* def rgb_tensor_render(obs, title='tensor_obs')
* def single_episodes(Env, args, verbose=True)
* def parallel_episodes(Env, args, verbose=False)
* def make_parallel_environments(Env, args)


# xml_files
* reacher
* humanoid


