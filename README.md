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

## Computation of Added Mass via mesh.
To compute the added mass terms of the Fossen model you can use Nemoh. Nemoh is an open-source Boundary Element Method (BEM) algorithm. To make the process easier both BEMRosetta and Matlab will be used.

First, download Nemoh via: https://lheea.ec-nantes.fr/valorisation/logiciels-et-brevets/nemoh-installation. You will need the executables version for the Win32 environment with the Matlab routines. Next, make sure you have Matlab installed, see https://nl.mathworks.com/help/install/install-products.html. 
To download BEMRosetta go to its Github page: https://github.com/BEMRosetta/BEMRosetta and download the BEMRosetta.exe from the /bin folder. 

Now we can do the mesh computations but first we need a mesh. Nemoh needs a specific type of mesh called a Nemoh.dat file. Luckily BEMRosetta can help us out in making and converting to that type of mesh. BEMRosetta can handle multiple types of input files but the one we are going to use here is the .stl type file. Both Solidworks and Blender can make a .stl file. So, make a 3D model of your robot in a programme of your choice. Make sure that it is as simple as possible, as the computations will take a very long time if there is a lot of detail in the mesh. So approximate cylinders by making them 8-sided and skip all the small nobs and dials etc. From this 3D model I have found it is easiest if you import it into Blender. In Blender make sure the scale of the robot is correct, and align all the axis correctly to the way you want your robot modelled. Also make sure to set the origin to the center of the robot. Nemoh is used for wave interaction computations, but as we do not want to compute those as the robot will be operating way below the surface and in environments without a lot of waves this needs to be corrected for. So far I have not found a way to disable the waves, but a trick to use is to set the robot to a depth of -50m in Blender. This way the waves will have diminishable effects on the robot. Now, export the model in Blender to a .stl file. 

In the > Mesh handling > load tab of BEMRosetta you can import this .stl file. In the > mesh handling > view tab click the arrow to the left on the right side of the screen. This reveals some tables with all the nodes and panels of the mesh. These usually need to be fixed: there can be some duplicates, or panels can be too small for nemoh to handle. This can be done via > mesh handling > process > mesh processing. Here click the 'simplify' and 'healing' buttons. Now we can export the mesh to a Nemoh.dat file. This can be done in > mesh handling > save as. For some reason Nemoh can only handle meshes that have a xz axis of symmetry if the mesh is a little bit complex. So, make sure to check the box of 'symmetry xz' otherwhise Nemoh will not work. Now you can convert it to a Nemoh.dat file. 

Now we can do the computation in Nemoh. First go to your > nemoh for matlab V x.xx > matlab routines folder in your hard drive. Here create a folder for your robot, say raybot_simple_v2. Inside of this folder you create two new folders. A 'mesh' and a 'results' folder. Place the .dat file of your robot mesh inside of the 'mesh' folder. 

Now we need to set up the computations. This is done via a Nemoh.cal file in the robot folder (raybot_simple_v2). It is easiest to copy paste this Nemoh.cal file from this repo as it is quite tricky to setup. Open the Nemoh.cal file and change the following lines:
line 9: change it to the location or your Nemoh.dat mesh file.
line 10: This is tricky, so take a look a the 'raybot_simple_v2' Nemoh.cal example file. This line needs the amount of nodes and panels of the mesh. The first part of the 'raybot_simple_v2' file has information on the nodes, the second part has information on the panels. The number of nodes are given in the first part in the second column, where they are counted. In this case the number of nodes is 540. (the row of zeroes doesn't count). The panels are a bit more trickier. Scroll all the way to the bottom the file and check the last line number with non-zero values. In this case this is 1460. Now we need to subtract (540+2) from it. (This is the line number at the bottom of the first part with zeroes). So: 1460-(540+2) = 918. This is the number of panels. These will sometimes be different from the number of nodes and panels in BEMRosetta so please use this cumbersome method. 
line 15,16,17,22,23,24: Here, set the z-axis for the computation of the moment forces to -50m (as you did in Blender). 
That's it! You can also change the density of the water etc but that should speak for itself. 

Now we need to change the 'ID.dat' file in the 'matlab routines' folder. Here, change the second row to the name of your robot folder, so raybot_simple_v2. The first line should be the amount of characters of the second row, so 16 in this case.

TODO: finish last part of readme for Nemoh
