bash
source ~/.bashrc 
isaaclab -p source/standalone/tutorials/04_sensors/run_ray_caster.py 
isaaclab -p source/standalone/tutorials/04_sensors/run_ray_caster.py --livestream 1
ls
ls logs/exit()
ls logs/exit
exit
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Robomaster-Direct-v0 --headless
exit
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Robomaster-Direct-v0 --headless --video --video_length 200
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Robomaster-Direct-v0 --headless
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Robomaster-Direct-v0 --headless --video --video_length 200
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Robomaster-Direct-v0 --headless
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Robomaster-Direct-v0 --headless --checkpoint logs/rl_games/robomaster_direct/2024-11-04_16-33-18/nn/robomaster_direct.pth 
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Robomaster-Direct-v0 --headless --video --video_length 200 --checkpoint logs/rl_games/robomaster_direct/2024-11-04_16-48-26/nn/robomaster_direct.pth 
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Robomaster-Direct-v0 --headless --video --video_length 200 --checkpoint logs/rl_games/robomaster_direct/2024-11-04_16-48-26/nn/robomaster_direct.pth --num_envs 32
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Robomaster-Direct-v0 --livestream 1 --checkpoint logs/rl_games/robomaster_direct/2024-11-04_16-48-26/nn/robomaster_direct.pth --num_envs 32
exit
code ~/.bashrc 
./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Robomaster-Glide-Direct-v0 --num_envs 1024
./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Robomaster-Glide-Direct-v0 --num_envs 1024 --livestream 1
./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Robomaster-Glide-Direct-v0 --num_envs 4
./isaaclab.sh -p source/standalone/environments/random_agent.py --task Isaac-Robomaster-Direct-v0 --num_envs 4 --livestream 1
