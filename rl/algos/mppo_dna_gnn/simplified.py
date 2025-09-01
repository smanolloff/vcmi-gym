def to_hdata(obs, done, links):
    device = obs.device
    res = HeteroData()
    res.obs = obs.unsqueeze(0)
    res.done = done.unsqueeze(0).float()
    res.value = torch.tensor(0., device=device)
    res.action = torch.tensor(0, device=device)
    res.reward = torch.tensor(0., device=device)
    res.logprob = torch.tensor(0., device=device)
    res.advantage = torch.tensor(0., device=device)
    res.ep_return = torch.tensor(0., device=device)

    res["hex"].x = obs[:STATE_SIZE_HEXES].view(165, STATE_SIZE_ONE_HEX)
    for lt in LINK_TYPES.keys():
        res["hex", lt, "hex"].edge_index = torch.as_tensor(links[lt]["index"], device=device)
        res["hex", lt, "hex"].edge_attr = torch.as_tensor(links[lt]["attrs"], device=device)

    return res


def to_hdata_batch(b_obs, b_done, tuple_links):
    b_hdatas = []
    for obs, done, links in zip(b_obs, b_done, tuple_links):
        b_hdatas.append(to_hdata(obs, done, links))
    return Batch.from_data_list(b_hdatas)


class Storage:
    def __init__(self, venv, num_vsteps, device):
        v = venv.num_envs
        self.rollout_buffer = []  # contains Batch() objects
        self.v_next_hdata = to_hdata_batch(
            torch.as_tensor(venv.reset()[0], device=device),
            torch.zeros(v, device=device),
            venv.call("links"),
        )

        self.bv_dones = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_values = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_rewards = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_advantages = torch.zeros((num_vsteps, venv.num_envs), device=device)
        self.bv_returns = torch.zeros((num_vsteps, venv.num_envs), device=device)


class MainAction(enum.IntEnum):
    WAIT = 0
    MOVE = enum.auto()
    AMOVE = enum.auto()
    SHOOT = enum.auto()


class ActionData:
    def __init__(self, act0, act0_dist, hex1, hex1_dist, hex2, hex2_dist, action):
        self.act0 = act0
        self.act0_dist = act0_dist
        self.act0_logprob = act0_dist.log_prob(act0)
        self.act0_entropy = act0_dist.entropy()

        self.hex1 = hex1
        self.hex1_dist = hex1_dist
        self.hex1_logprob = hex1_dist.log_prob(hex1)
        self.hex1_entropy = hex1_dist.entropy()

        self.hex2 = hex2
        self.hex2_dist = hex2_dist
        self.hex2_logprob = hex2_dist.log_prob(hex2)
        self.hex2_entropy = hex2_dist.entropy()

        self.action = action
        self.logprob = self.act0_logprob + self.hex1_logprob + self.hex2_logprob
        self.entropy = self.act0_entropy + self.hex1_entropy + self.hex2_entropy


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        link_types = [
            "ADJACENT",
            "REACH",
            "RANGED_MOD",
            "ACTS_BEFORE",
            "MELEE_DMG_REL",
            "RETAL_DMG_REL",
            "RANGED_DMG_REL"
        ]

        gatconv_kwargs = dict(
            in_channels=(-1, -1),
            out_channels=config["gnn_z_size"],
            heads=config["gnn_heads"],
            add_self_loops=True
        )

        self.layers = nn.ModuleList()
        for _ in range(config["gnn_layers"]):
            layer = dict()
            for lt in link_types:
                layer[("hex", lt, "hex")] = gnn.GATConv(**gatconv_kwargs)
                # XXX: a leaky_relu is applied after each GATConv, see encode()
            self.layers.append(gnn.HeteroConv(layer))

        self.encoder_merged = nn.Sequential(
            nn.LazyLinear(config["z_size_merged"]),
            nn.LeakyReLU()
        )

        self.actor = nn.LazyLinear(len(MainAction)+165+165)
        self.critic = nn.LazyLinear(1)

        # Init lazy layers (must be before weight/bias init)
        with torch.no_grad():
            obs = torch.randn([2, STATE_SIZE])
            done = torch.zeros(2)
            links = 2 * [VcmiEnv.OBSERVATION_SPACE["links"].sample()]
            hdata = to_hdata_batch(obs, done, links)
            z = self.encode(hdata)
            self._get_actdata_eval(z, obs)
            self._get_value(z)

        def kaiming_init(linlayer):
            # Assume LeakyReLU's negative slope is the default
            a = torch.nn.LeakyReLU().negative_slope
            nn.init.kaiming_uniform_(linlayer.weight, nonlinearity='leaky_relu', a=a)
            nn.init.zeros_(linlayer.bias)

        def xavier_init(linlayer):
            nn.init.xavier_uniform_(linlayer.weight)
            nn.init.zeros_(linlayer.bias)

        kaiming_init(self.encoder_merged[0])
        xavier_init(self.actor)
        xavier_init(self.critic)

    def encode(self, hdata):
        x_dict = hdata.x_dict

        for layer in self.layers:
            x_dict = layer(x_dict, hdata.edge_index_dict, edge_attr_dict=hdata.edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        zhex, hmask = to_dense_batch(x_dict["hex"], hdata["hex"].batch)
        assert torch.all(hmask)
        return self.encoder_merged(zhex.flatten(start_dim=1))

    def _get_value(self, z):
        return self.critic(z)

    def _get_actdata_train(self, z_merged, obs, action):
        act0, hex1, hex2 = self.inverse_table[action].unbind(1)

        # ... logic for calculating the distribution of the 3 action components
        # (act0, hex1, hex2)

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    def _get_actdata_eval(self, z_merged, obs):
        # ... logic for calculating the distribution of the 3 action components
        # (act0, hex1, hex2)

        act0 = dist_act0.sample()
        hex1 = dist_hex1.sample()
        hex2 = dist_hex2.sample()
        action = consolidate_action(act0, hex1, hex2)

        return ActionData(
            act0=act0, act0_dist=dist_act0,
            hex1=hex1, hex1_dist=dist_hex1,
            hex2=hex2, hex2_dist=dist_hex2,
            action=action,
        )

    def get_actdata_train(self, hdata):
        z_merged = self.encode(hdata)
        return self._get_actdata_train(z_merged, hdata.obs, hdata.action)

    def get_actdata_eval(self, hdata):
        z_merged = self.encode(hdata)
        return self._get_actdata_eval(z_merged, hdata.obs)

    def get_value(self, hdata):
        z_merged = self.encode(hdata)
        return self._get_value(z_merged)


class DNAModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model_policy = Model(config)
        self.model_value = Model(config)
        self.device = device
        self.to(device)


def collect_samples(logger, model, venv, num_vsteps, storage):
    assert not torch.is_inference_mode_enabled()  # causes issues during training
    assert not torch.is_grad_enabled()

    device = model.device

    storage.rollout_buffer.clear()

    for vstep in range(num_vsteps):
        logger.debug("(train) vstep: %d" % vstep)

        v_hdata = storage.v_next_hdata

        v_actdata = model.model_policy.get_actdata_eval(v_hdata)
        v_value = model.model_value.get_value(v_hdata)

        v_hdata.action[:] = v_actdata.action
        v_hdata.logprob[:] = v_actdata.logprob
        v_hdata.value[:] = v_value.flatten()

        v_obs, v_rew, v_term, v_trunc, v_info = venv.step(v_actdata.action.cpu().numpy())

        v_hdata.reward[:] = torch.as_tensor(v_rew, device=device)

        storage.bv_dones[vstep] = v_hdata.done
        storage.bv_values[vstep] = v_hdata.value
        storage.bv_rewards[vstep] = v_hdata.reward
        storage.v_next_hdata = to_hdata_batch(
            torch.as_tensor(v_obs, device=device),
            torch.as_tensor(np.logical_or(v_term, v_trunc), device=device),
            venv.call("links")
        )

        storage.rollout_buffer.append(v_hdata)

    assert len(storage.rollout_buffer) == num_vsteps

    # bootstrap value if not done
    v_next_value = model.model_value.get_value(storage.v_next_hdata).flatten()
    storage.v_next_hdata.value[:] = v_next_value


def train_model(
    logger,
    model,
    optimizer_policy,
    optimizer_value,
    optimizer_distill,
    autocast_ctx,
    scaler,
    storage,
    train_config
):
    assert torch.is_grad_enabled()

    num_vsteps = train_config["num_vsteps"]
    num_envs = train_config["env"]["num_envs"]

    # # compute advantages
    with torch.no_grad():
        lastgaelam = 0

        for t in reversed(range(num_vsteps)):
            if t == num_vsteps - 1:
                nextnonterminal = 1.0 - storage.v_next_hdata.done
                nextvalues = storage.v_next_hdata.value
            else:
                nextnonterminal = 1.0 - storage.bv_dones[t + 1]
                nextvalues = storage.bv_values[t + 1]
            delta = storage.bv_rewards[t] + train_config["gamma"] * nextvalues * nextnonterminal - storage.bv_values[t]
            storage.bv_advantages[t] = lastgaelam = delta + train_config["gamma"] * train_config["gae_lambda"] * nextnonterminal * lastgaelam
        storage.bv_returns[:] = storage.bv_advantages + storage.bv_values

        for b in range(num_vsteps):
            v_hdata = storage.rollout_buffer[b]
            v_hdata.advantage[:] = storage.bv_advantages[b]
            v_hdata.ep_return[:] = storage.bv_returns[b]

    batch_size = num_vsteps * num_envs
    minibatch_size = int(batch_size // train_config["num_minibatches"])

    # Explode buffer into individual hdatas (dataloader forms a single, large batch)
    dataloader = DataLoader(
        [hdata for batch in storage.rollout_buffer for hdata in batch.to_data_list()],
        batch_size=minibatch_size,
        shuffle=True
    )

    clipfracs = []

    policy_losses = torch.zeros(train_config["num_minibatches"])
    entropy_losses = torch.zeros(train_config["num_minibatches"])
    value_losses = torch.zeros(train_config["num_minibatches"])
    distill_losses = torch.zeros(train_config["num_minibatches"])

    for epoch in range(train_config["update_epochs"]):
        for i, mb in enumerate(dataloader):
            newactdata = model.model_policy.get_actdata_train(mb)

            logratio = newactdata.logprob - mb.logprob
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > train_config["clip_coef"]).float().mean().item()]

            if train_config["norm_adv"]:
                # The 1e-8 is not numerically safe under autocast
                # mb_advantages = (mb.advantage - mb.advantage.mean()) / (mb.advantage.std() + 1e-8)
                with autocast_ctx(False):
                    adv32 = mb_advantages.float()
                    mean = adv32.mean()
                    var = adv32.var(unbiased=False)
                    norm = (adv32 - mean) * torch.rsqrt(var + 1e-8)
            mb_advantages = norm.to(adv.dtype)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_config["clip_coef"], 1 + train_config["clip_coef"])
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = newactdata.entropy.mean()

            policy_losses[i] = policy_loss.detach()
            entropy_losses[i] = entropy_loss.detach()

            action_loss = policy_loss - entropy_loss * train_config["ent_coef"]

            with autocast_ctx(False):
                scaler.scale(action_loss).backward()
                scaler.unscale_(optimizer_policy)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_policy.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_policy)
                scaler.update()
                optimizer_policy.zero_grad()

        if train_config["target_kl"] is not None and approx_kl > train_config["target_kl"]:
            break

    # Value network optimization
    for epoch in range(train_config["update_epochs"]):
        for i, mb in enumerate(dataloader):
            newvalue = model.model_value.get_value(mb)

            # Value loss
            newvalue = newvalue.view(-1)
            if train_config["clip_vloss"]:
                v_loss_unclipped = (newvalue - mb.ep_return) ** 2
                v_clipped = mb.value + torch.clamp(
                    newvalue - mb.value,
                    -train_config["clip_coef"],
                    train_config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - mb.ep_return) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()
            else:
                # XXX: SIMO: SB3 does not multiply by 0.5 here
                value_loss = 0.5 * ((newvalue - mb.ep_return) ** 2).mean()

            value_losses[i] = value_loss.detach()

            with autocast_ctx(False):
                scaler.scale(value_loss).backward()
                scaler.unscale_(optimizer_value)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_value.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_value)
                scaler.update()
                optimizer_value.zero_grad()

    # Value network to policy network distillation
    model.model_policy.zero_grad(True)  # don't clone gradients
    old_model_policy = copy.deepcopy(model.model_policy).to(model.device)
    old_model_policy.eval()
    for epoch in range(train_config["update_epochs"]):
        for i, mb in enumerate(dataloader):
            # Compute policy and value targets
            with torch.no_grad():
                old_actdata = old_model_policy.get_actdata_eval(mb)
                value_target = model.model_value.get_value(mb)

            # XXX: must pass action=<old_action> to ensure masks for hex1 and hex2 are the same
            #     (if actions differ, masks will differ and KLD will become NaN)
            new_z = model.model_policy.encode(mb)
            new_actdata = model.model_policy._get_actdata_train(new_z, mb.obs, mb.action)
            new_value = model.model_policy._get_value(new_z)

            # Distillation loss
            distill_actloss = (
                kld(old_actdata.act0_dist, new_actdata.act0_dist)
                + kld(old_actdata.hex1_dist, new_actdata.hex1_dist)
                + kld(old_actdata.hex2_dist, new_actdata.hex2_dist)
            ).mean()

            distill_vloss = 0.5 * (new_value.view(-1) - value_target).square().mean()
            distill_loss = distill_vloss + train_config["distill_beta"] * distill_actloss

            distill_losses[i] = distill_loss.detach()

            with autocast_ctx(False):
                scaler.scale(distill_loss).backward()
                scaler.unscale_(optimizer_distill)  # needed for clip_grad_norm
                nn.utils.clip_grad_norm_(model.model_policy.parameters(), train_config["max_grad_norm"])
                scaler.step(optimizer_distill)
                scaler.update()
                optimizer_distill.zero_grad()

    y_pred, y_true = storage.bv_values.cpu().numpy(), storage.bv_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def main(config):
    train_config = dig(config, "train")
    learning_rate = config["train"]["learning_rate"]

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    train_venv = create_venv(train_config["env"]["kwargs"], train_config["env"]["num_envs"], sync=train_config["env"].get("sync", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = train_config["env"]["num_envs"]
    num_steps = train_config["num_vsteps"] * num_envs
    batch_size = int(num_steps)
    assert batch_size % train_config["num_minibatches"] == 0, f"{batch_size} % {train_config['num_minibatches']} == 0"
    storage = Storage(train_venv, train_config["num_vsteps"], device)

    model = DNAModel(config=config["model"], device=device)

    optimizer_policy = torch.optim.Adam(model.model_policy.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(model.model_value.parameters(), lr=learning_rate)
    optimizer_distill = torch.optim.Adam(model.model_policy.parameters(), lr=learning_rate)

    # No-op autocast and scaler
    autocast_ctx = contextlib.nullcontext
    scaler = torch.GradScaler(device.type, enabled=False)

    try:
        while True:
            with timers["sample"], torch.no_grad(), autocast_ctx(True):
                model.eval()
                collect_samples(
                    logger=logger,
                    model=model,
                    venv=train_venv,
                    num_vsteps=train_config["num_vsteps"],
                    storage=storage,
                )

            model.train()
            with timers["train"], autocast_ctx(True):
                train_model(
                    logger=logger,
                    model=model,
                    optimizer_policy=optimizer_policy,
                    optimizer_value=optimizer_value,
                    optimizer_distill=optimizer_distill,
                    autocast_ctx=autocast_ctx,
                    scaler=scaler,
                    storage=storage,
                    train_config=train_config,
                )
