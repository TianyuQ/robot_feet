cd /home/tq877/Tianyu/robot_feet/robot_feet/learning/scripts

# Example: stronger (typical default-like)
python3 t1_joystick_flat_terrain_force.py --train --force-impulse-penalty-scale -0.1

python3 t1_joystick_flat_terrain_force.py --train --force-impulse-penalty-scale -0.3

python3 t1_joystick_flat_terrain_force.py --eval --plot --load-checkpoint-path robot_feet/learning/scripts/logs/T1ForceJoystickFlatTerrain-Fimp-0.1-steps200000000/checkpoints --x-vel 0.5 --y-vel 0 --yaw-vel 0

python3 t1_joystick_flat_terrain_force.py --eval --plot --load-checkpoint-path robot_feet/learning/scripts/logs/T1ForceJoystickFlatTerrain-Fimp-0.3-steps200000000/checkpoints --x-vel 0.5 --y-vel 0 --yaw-vel 0