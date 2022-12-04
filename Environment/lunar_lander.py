# install gym, box2d-py, and keyboard

import gym
import keyboard

env = gym.make(
   "LunarLander-v2",
   render_mode="human",
   gravity=-2.0,
)
observation, info = env.reset(seed=42)

while True:
#for _ in range(1000):
   
   if keyboard.is_pressed('esc'):
      break

   action = 0

   if keyboard.is_pressed('up arrow') or keyboard.is_pressed('down arrow'):
      action = 2
   if keyboard.is_pressed('left arrow') :
      action = 1
   if keyboard.is_pressed('right arrow') :
      action = 3

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()