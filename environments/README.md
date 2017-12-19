#  Todo


NOW:

Torso!

angle


# Environment


In my custom classes i load the xml file for the robot only once. Only if no scene has been initialized. For me this makes the reset() work as expected.

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

I do not have great knowledge about everything in the bullet/gym code but it seems superfluous to load the xml file each reset. At least in the singleplayer scenario, I have not tried multiplayer.

## gym_env.py

MyGymEnv is a class which is the wrapper for OpenAI gym. It contains important functions such as:
* _seed
* _reset
* _render
* _step


