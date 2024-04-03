from pettingzoo.atari import boxing_v2


# there are 4 different groups in total
GROUP_NUMBER = 4
# agent number in each group
AGENT_NUMBER = 1

rom_path = '/usr/local/lib/python3.7/dist-packages/AutoROM/roms'

env = boxing_v2.parallel_env(render_mode="human", auto_rom_install_path=rom_path)
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
