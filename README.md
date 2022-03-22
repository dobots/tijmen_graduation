# Tijmen Graduation

 :sparkles: Welcome to the most beautiful repo at Dobots :sparkles: 

First, follow steps 1-4 at https://github.com/Liquid-ai/Plankton 

Then, to run the simulation, run
```sh
ros2 launch uuv_gazebo_worlds ocean_waves.launch
ros2 launch raybot_description upload_raybot.launch
ros2 launch uuv_control_cascaded_pid joy_velocity.launch uuv_name:=raybot model_name:=raybot joy_id:=1
```
For the joy_id, check how to find your own joy id at https://github.com/Liquid-ai/Plankton step 5. 
