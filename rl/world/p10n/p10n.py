import torch
import torch.nn as nn
import math
import enum
import contextlib

from torch.nn.functional import cross_entropy

from ..util.buffer_base import BufferBase
from ..util.dataset_vcmi import Data, Context
from ..util.misc import layer_init
from ..util.obs_index import ObsIndex
from ..util.timer import Timer

from ..util.constants_v11 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    HEX_ACT_MAP,
    N_ACTIONS,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


class MainAction(enum.IntEnum):
    RESET = 0
    WAIT = enum.auto()
    HEX = enum.auto()


class Prediction(enum.IntEnum):
    PROBS = 0               # softmax(logits)
    SAMPLE = enum.auto()    # sample(softmax(logits))
    GREEDY = enum.auto()    # argmax(logits)


def vcmi_dataloader_functor():
    def mw(data: Data, ctx: Context):
        # First transition is OUR state (the action is random for it)
        # Last transition is OUR state (the action is always -1 there)
        # => skip unless state is terminal (action=-1 stands for done=true)
        if ctx.transition_id == 0 or (ctx.transition_id == ctx.num_transitions - 1 and not data.done):
            return None

        return data
    return mw


def s3_dataloader_functor():
    # XXX: must only pass through non-terminal obs which are from the ENEMY
    #      Below is the code from v10, should be *much* simpler in v11
    #      (as there is now a BATTLE_SIDE_ACTIVE_PLAYER global attribute)
    #
    # hexes = obs[STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER:].reshape(165, -1)
    # index_is_active = HEX_ATTR_MAP["STACK_QUEUE"][1]  # 1 if first in queue (zero null => index=offset)
    # # index_is_blue = np.arange(HEX_ATTR_MAP["STACK_SIDE"][1] + HEX_ATTR_MAP["STACK_SIDE"][2]] + 2  # 1 if blue ([null, red, blue] => index=offset+2)
    # extracted = hexes[:, [index_is_active, index_is_blue]]
    # # => (165, 2), where 2 = [is_active, is_blue]
    # mask = extracted[..., 0] != 0
    # # => 165 bools (True for hexes for which is_active=True)
    # idx = np.argmax(mask)  # index of 1st hex with active stack on it
    # is_blue = extracted[idx, 1]  # is_blue for that hex
    # if is_blue:
    #     index_battle_in_progress = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1]  # 1 if winner is N/A (explicit null => index=offset)
    #     if action > 0:
    #         assert obs[index_battle_in_progress] == 1, obs[index_battle_in_progress]
    #         with self.timer_idle:
    #             yield obs, action
    raise NotImplementedError()

    # def mw(data: Data, ctx: Context):
    #     if ctx.transition_id == ctx.num_transitions - 1:
    #         state["reward_carry"] = data.reward
    #         if not data.done:
    #             return None
    #     if ctx.transition_id == 0 and ctx.ep_steps > 0:
    #         return data._replace(reward=state["reward_carry"])
    #     return data

    # return mw


class Buffer(BufferBase):
    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # XXX: ALL indices are valid:
        # (terminal obs has action = -1 which is used for `done` prediction)
        sampled_indices = torch.randint(max_index, (batch_size,), device=self.device)
        obs = self.containers["obs"][sampled_indices]
        mask = self.containers["mask"][sampled_indices]
        reward = self.containers["reward"][sampled_indices]
        done = self.containers["done"][sampled_indices]
        action = self.containers["action"][sampled_indices]

        return obs, mask, reward, done, action

    def sample_iter(self, batch_size):
        max_index = self.capacity if self.full else self.index

        shuffled_indices = torch.randperm(max_index, device=self.device)

        # will fail if self.full is false
        assert len(shuffled_indices) == self.capacity

        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]

            # NOTE: reward is the result of prev action (see vcmidataset.py)
            #  XXX: reward prediction gets tricky for step0 (no reward!)
            yield (
                self.containers["obs"][batch_indices],
                self.containers["mask"][batch_indices],
                self.containers["reward"][batch_indices],
                self.containers["done"][batch_indices],
                self.containers["action"][batch_indices],
            )


class ActionPredictionModel(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        obsind = ObsIndex(device)

        self.abs_index = obsind.abs_index
        self.rel_index_global = obsind.rel_index_global
        self.rel_index_player = obsind.rel_index_player
        self.rel_index_hex = obsind.rel_index_hex

        # See notes in t10n/main.py
        emb_calc = lambda n: math.ceil(math.sqrt(n))

        self.encoder_action = nn.Embedding(N_ACTIONS, emb_calc(N_ACTIONS))

        #
        # Global encoders
        #

        # Continuous:
        # (B, n)
        self.encoder_global_continuous = nn.Identity()

        # Binaries:
        # (B, n)
        self.encoder_global_binary = nn.Identity()
        if self.abs_index["global"]["binary"].numel():
            n_binary_features = len(self.abs_index["global"]["binary"])
            self.encoder_global_binary = nn.LazyLinear(n_binary_features)
            # No nonlinearity needed

        # Categoricals:
        # [(B, C1), (B, C2), ...]
        self.encoders_global_categoricals = nn.ModuleList([])
        for ind in self.rel_index_global["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_global_categoricals.append(cat_emb_size)

        # Merge
        z_size_global = 128
        self.encoder_merged_global = nn.Sequential(
            # => (B, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS)
            nn.LazyLinear(z_size_global),
            nn.LeakyReLU(),
        )
        # => (B, Z_GLOBAL)

        #
        # Player encoders
        #

        # Continuous per player:
        # (B, n)
        self.encoder_player_continuous = nn.Identity()

        # Binaries per player:
        # (B, n)
        self.encoder_player_binary = nn.Identity()
        if self.abs_index["player"]["binary"].numel():
            n_binary_features = len(self.abs_index["player"]["binary"][0])
            self.encoder_player_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per player:
        # [(B, C1), (B, C2), ...]
        self.encoders_player_categoricals = nn.ModuleList([])
        for ind in self.rel_index_player["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_player_categoricals.append(cat_emb_size)

        # Merge per player
        z_size_player = 128
        self.encoder_merged_player = nn.Sequential(
            # => (B, 2, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS)
            nn.LazyLinear(z_size_player),
            nn.LeakyReLU(),
        )
        # => (B, 2, Z_PLAYER)

        #
        # Hex encoders
        #

        # Continuous per hex:
        # (B, n)
        self.encoder_hex_continuous = nn.Identity()

        # Binaries per hex:
        # (B, n)
        if self.abs_index["hex"]["binary"].numel():
            n_binary_features = len(self.abs_index["hex"]["binary"][0])
            self.encoder_hex_binary = nn.LazyLinear(n_binary_features)

        # Categoricals per hex:
        # [(B, C1), (B, C2), ...]
        self.encoders_hex_categoricals = nn.ModuleList([])
        for ind in self.rel_index_hex["categoricals"]:
            cat_emb_size = nn.Embedding(num_embeddings=len(ind), embedding_dim=emb_calc(len(ind)))
            self.encoders_hex_categoricals.append(cat_emb_size)

        # Merge per hex
        z_size_hex = 512
        self.encoder_merged_hex = nn.Sequential(
            # => (B, 165, N_ACTIONS + N_CONT_FEATS + N_BIN_FEATS + C*N_CAT_FEATS)
            nn.LazyLinear(z_size_hex),
            nn.LeakyReLU(),
        )
        # => (B, 165, Z_HEX)

        # Transformer (hexes only)
        self.transformer_hex = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=z_size_hex, nhead=8, batch_first=True),
            num_layers=6
        )
        # => (B, 165, Z_HEX)

        #
        # Aggregator
        #

        # (B, Z_GLOBAL + AVG(2*Z_PLAYER) + AVG(165*Z_HEX))
        self.aggregator = nn.LazyLinear(2048)
        # => (B, Z_AGG)

        #
        # Heads
        #

        # => (B, Z_AGG)
        self.head_main = nn.LazyLinear(len(MainAction))
        # => (B, MainAction._count)

        # => (B, 165, Z_AGG + Z_HEX)
        self.head_hex = nn.LazyLinear(1 + len(HEX_ACT_MAP))
        # => (B, 165, hex_score + HexAction._count)

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            obs = torch.randn([2, DIM_OBS], device=device)
            self.forward(obs)

        layer_init(self)

    def forward(self, obs):
        assert obs.device.type == self.device.type, f"{obs.device.type} == {self.device.type}"

        global_continuous_in = obs[:, self.abs_index["global"]["continuous"]]
        global_binary_in = obs[:, self.abs_index["global"]["binary"]]
        global_categorical_ins = [obs[:, ind] for ind in self.abs_index["global"]["categoricals"]]
        global_continuous_z = self.encoder_global_continuous(global_continuous_in)
        global_binary_z = self.encoder_global_binary(global_binary_in)

        # XXX: Embedding layers expect single-integer inputs
        #      e.g. for input with num_classes=4, instead of `[0,0,1,0]` it expects just `2`
        global_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_global_categoricals, global_categorical_ins)], dim=-1)
        global_merged = torch.cat((global_continuous_z, global_binary_z, global_categorical_z), dim=-1)
        z_global = self.encoder_merged_global(global_merged)
        # => (B, Z_GLOBAL)

        player_continuous_in = obs[:, self.abs_index["player"]["continuous"]]
        player_binary_in = obs[:, self.abs_index["player"]["binary"]]
        player_categorical_ins = [obs[:, ind] for ind in self.abs_index["player"]["categoricals"]]
        player_continuous_z = self.encoder_player_continuous(player_continuous_in)
        player_binary_z = self.encoder_player_binary(player_binary_in)
        player_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_player_categoricals, player_categorical_ins)], dim=-1)
        player_merged = torch.cat((player_continuous_z, player_binary_z, player_categorical_z), dim=-1)
        z_player = self.encoder_merged_player(player_merged)
        # => (B, 2, Z_PLAYER)

        hex_continuous_in = obs[:, self.abs_index["hex"]["continuous"]]
        hex_binary_in = obs[:, self.abs_index["hex"]["binary"]]
        hex_categorical_ins = [obs[:, ind] for ind in self.abs_index["hex"]["categoricals"]]
        hex_continuous_z = self.encoder_hex_continuous(hex_continuous_in)
        hex_binary_z = self.encoder_hex_binary(hex_binary_in)
        hex_categorical_z = torch.cat([enc(x.argmax(dim=-1)) for enc, x in zip(self.encoders_hex_categoricals, hex_categorical_ins)], dim=-1)
        hex_merged = torch.cat((hex_continuous_z, hex_binary_z, hex_categorical_z), dim=-1)
        z_hex = self.encoder_merged_hex(hex_merged)
        z_hex = self.transformer_hex(z_hex)
        # => (B, 165, Z_HEX)

        mean_z_player = z_player.mean(dim=1)
        mean_z_hex = z_hex.mean(dim=1)
        z_agg = self.aggregator(torch.cat([z_global, mean_z_player, mean_z_hex], dim=-1))
        # => (B, Z_AGG)

        #
        # Outputs
        #

        main_out = self.head_main(z_agg)
        # => (B, 3)

        hex_out = self.head_hex(torch.cat([z_agg.unsqueeze(1).expand(-1, 165, -1), z_hex], dim=-1))
        # => (B, 165, HexActions + hex_score)  # score=choose this hex

        return main_out, hex_out

    def _predict_probs(self, obs, logits):
        main_logits, hex_logits = logits
        B = obs.shape[0]

        probs_main_action = main_logits.softmax(dim=-1)
        # => (B, len(MainAction)) of MainAction probs

        probs_hex_id = hex_logits[:, :, 0].softmax(dim=-1)
        # => (B, 165) of Hex id probs

        probs_hex_action = hex_logits[:, :, 1:].softmax(dim=-1)
        # => (B, 165, len(HexAction)) of hex actions probs for the each hex id

        # Now add the chance of choosing each hex:
        probs_hex_action = probs_hex_action * probs_hex_id.reshape(B, 165, 1).expand(probs_hex_action.shape)
        # => (B, 165, len(HexAction))

        # Now add the chance of choosing a hex action at all:
        probs_hex_action = probs_hex_action * probs_main_action[:, MainAction.HEX].reshape(B, 1, 1).expand(probs_hex_action.shape)
        # => (B, 165, len(HexAction))

        probs = torch.cat((
            probs_main_action[:, MainAction.RESET].reshape(B, 1),
            probs_main_action[:, MainAction.WAIT].reshape(B, 1),
            probs_hex_action.flatten(start_dim=1)
        ), dim=1)
        # => (B, 2312)

        # assert probs.sum(dim=1) == 1, probs.sum()
        return probs

    def _predict_action(self, obs, logits, strategy):
        main_logits, hex_logits = logits

        if strategy == Prediction.GREEDY:
            def pick(x):
                return x.argmax(dim=1)
        else:
            def pick(x):
                num_classes = x.shape[-1]
                probs_2d = x.softmax(dim=-1).view(-1, num_classes)
                return torch.multinomial(probs_2d, num_samples=1).view(x.shape[:-1])

        action_main = pick(main_logits)
        # => (B) of MainAction values

        hex_id = pick(hex_logits[:, :, 0])
        # => (B) of Hex ids

        # arange as the B index since hex_id contains B values (ranging 0..164)
        hex_action = pick(hex_logits[torch.arange(hex_id.shape[0]), hex_id, 1:])
        # => (B) of hex actions for the predicted hex ids

        # Action = 2 + hex_id * N_ACTIONS + hex_action
        action_calc = 2 + hex_id * len(HEX_ACT_MAP) + hex_action

        return (
            action_main
            .where(action_main != MainAction.RESET, -1)         # RESET => -1 (e.g. done=True)
            .where(action_main != MainAction.WAIT, 1)           # WAIT => 1
            .where(action_main != MainAction.HEX, action_calc)  # HEX => calc
        )

    def predict_(self, obs, logits=None, strategy=Prediction.GREEDY):
        if logits is None:
            logits = self.forward(obs)

        if strategy == Prediction.PROBS:
            return self._predict_probs(obs, logits)
        else:
            return self._predict_action(obs, logits, strategy)

    def predict(self, obs, strategy=Prediction.GREEDY):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.predict_(obs, strategy=strategy)[0].item()


def compute_loss(action, logits_main, logits_hex):
    assert all(action != 0), "Found retreat action: %s" % action

    # self.where(condition, y) is equivalent to torch.where(condition, self, y)
    target_main = (
        action
        .where(action != -1, MainAction.RESET)  # -1 => RESET (e.g. done=True)
        .where(action != 1, MainAction.WAIT)    # 1 => WAIT
        .where(action < 2, MainAction.HEX)      # 2+ => HEX
    )

    #
    # Loss for MAIN action
    #

    loss_main = cross_entropy(logits_main, target_main)

    #
    # Loss for HEX actions (if any)
    #

    # Get relevant indexes of actions what are on a hex
    # (B) of actions
    actions_on_hex = torch.where(action >= 2)[0]
    # => (B') of indexes

    # Guard against NaN losses
    if actions_on_hex.numel() == 0:
        return loss_main, torch.tensor(0.), torch.tensor(0.)

    # logits_hex is (B, 165, 1 + N_HEX_ACTIONS)
    # The CE logits will be the score logit (1st logit) of each hex
    # The CE target will be the real, numeric hex id
    # i.e. form a single one-hot categorical from the 165 score logits
    #      and compare it against a numeric hex ID
    logits_hex_score = logits_hex[actions_on_hex, :, 0]
    # => (B', 165) of score logits
    target_hex_id = (action[actions_on_hex] - 2) // len(HEX_ACT_MAP)
    # (B') of hex_ids
    loss_hex = cross_entropy(logits_hex_score, target_hex_id)

    # logits_hex is (B, 165, 1 + N_HEX_ACTIONS)
    # The CE logits will be the hexaction logits (>1st logit) of the target hex
    # The CE target will be the real, numeric hex action for the target hex
    logits_hex_action = logits_hex[actions_on_hex, target_hex_id, 1:]
    # => (B', N_HEX_ACTIONS) of hexaction logits for the target hex
    target_hex_action = (action[actions_on_hex] - 2) % len(HEX_ACT_MAP)
    # (B') of hex_actions
    loss_hexaction = cross_entropy(logits_hex_action, target_hex_action)

    return loss_main, loss_hex, loss_hexaction


def train_model(
    logger,
    model,
    optimizer,
    scaler,
    buffer,
    stats,
    loss_weights,
    epochs,
    batch_size,
    accumulate_grad,
    wlog
):
    model.train()
    main_losses = []
    hex_losses = []
    hexaction_losses = []
    total_losses = []
    timer = Timer()

    maybe_autocast = torch.amp.autocast(model.device.type) if scaler else contextlib.nullcontext()

    assert buffer.capacity % batch_size == 0, f"{buffer.capacity} % {batch_size} == 0"

    if accumulate_grad:
        grad_steps = buffer.capacity // batch_size
        assert grad_steps > 0

    for epoch in range(epochs):
        timer.start()
        for batch in buffer.sample_iter(batch_size):
            timer.stop()
            obs, mask, rew, done, action = batch

            with maybe_autocast:
                pred_logits_main, pred_logits_hex = model(obs)
                loss_main, loss_hex, loss_hexaction = compute_loss(action, pred_logits_main, pred_logits_hex)
                loss_tot = loss_main + loss_hex + loss_hexaction

            main_losses.append(loss_main.item())
            hex_losses.append(loss_hex.item())
            hexaction_losses.append(loss_hexaction.item())
            total_losses.append(loss_tot.item())

            if accumulate_grad:
                if scaler:
                    scaler.scale(loss_tot / grad_steps).backward()
                else:
                    (loss_tot / grad_steps).backward()
            else:
                if scaler:
                    scaler.scale(loss_tot).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_tot.backward()
                    optimizer.step()
                optimizer.zero_grad()
            timer.start()
        timer.stop()

        if accumulate_grad:
            # assert grad_step == 0, "Sample waste: %d sample batches"
            # Update once after the entire buffer is exhausted
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    main_loss = sum(main_losses) / len(main_losses)
    hex_loss = sum(hex_losses) / len(hex_losses)
    hexaction_loss = sum(hexaction_losses) / len(hexaction_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()

    wlog["train_loss/main"] = main_loss
    wlog["train_loss/hex"] = hex_loss
    wlog["train_loss/hexaction"] = hexaction_loss
    wlog["train_loss/total"] = total_loss
    wlog["train_dataset/wait_time_s"] = total_wait

    return total_loss


def eval_model(
    logger,
    model,
    loss_weights,
    buffer,
    batch_size,
    wlog
):
    model.eval()

    main_losses = []
    hex_losses = []
    hexaction_losses = []
    total_losses = []
    match_ratios = []
    timer = Timer()

    timer.start()
    for batch in buffer.sample_iter(batch_size):
        timer.stop()
        obs, mask, rew, done, action = batch

        with torch.no_grad():
            pred_logits_main, pred_logits_hex = model(obs)
            pred_action = model.predict_(obs, (pred_logits_main, pred_logits_hex))
            match_ratios.append((pred_action == action).float().mean().item())

        loss_main, loss_hex, loss_hexaction = compute_loss(action, pred_logits_main, pred_logits_hex)
        loss_tot = loss_main + loss_hex + loss_hexaction

        main_losses.append(loss_main.item())
        hex_losses.append(loss_hex.item())
        hexaction_losses.append(loss_hexaction.item())
        total_losses.append(loss_tot.item())

        timer.start()
    timer.stop()

    main_loss = sum(main_losses) / len(main_losses)
    hex_loss = sum(hex_losses) / len(hex_losses)
    hexaction_loss = sum(hexaction_losses) / len(hexaction_losses)
    total_loss = sum(total_losses) / len(total_losses)
    total_wait = timer.peek()
    match_ratio = sum(match_ratios) / len(match_ratios)

    wlog["eval_loss/main"] = main_loss
    wlog["eval_loss/hex"] = hex_loss
    wlog["eval_loss/hexaction"] = hexaction_loss
    wlog["eval_loss/total"] = total_loss
    wlog["eval/greedy_match_ratio"] = match_ratio
    wlog["eval_dataset/wait_time_s"] = total_wait

    return total_loss
