import vcmi_gym

if __name__ == '__main__':
    env = vcmi_gym.VcmiEnv("pikemen.vmap")
    print(env.render())
    env.step(10)
    print(env.render())
