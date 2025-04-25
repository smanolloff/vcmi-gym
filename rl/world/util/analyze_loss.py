import torch

from ..util.constants_v12 import GLOBAL_ATTR_MAP, PLAYER_ATTR_MAP, HEX_ATTR_MAP


#
# Returns three dicts:
#
#   1/ Dict with structure:
#
#       {
#           "global": [
#               ("GLOBAL_ATTR1", tensor([diff_frac, diff_loss, eval_loss, train_loss]),
#               ("GLOBAL_ATTR2", tensor([diff_frac, diff_loss, eval_loss, train_loss]),
#               ...
#           ],
#           "player": [ ... ]
#           "hex": [ ... ]
#       }
#
#   2/ Same as 1/, but each array has contains the topN attrs by diff_frac
#   3/ Same as 1/, but each array has contains the topN attrs by diff_loss
#
def analyze_loss(model, train_attrlosses, eval_attrlosses, topk=3, verbose=False):
    _print = lambda msg: print(msg) if verbose else None

    attrnames = {"global": list(GLOBAL_ATTR_MAP), "player": list(PLAYER_ATTR_MAP), "hex": list(HEX_ATTR_MAP)}
    attrstats = {
        # 4 = diff_frac, diff_loss, eval_loss, train_loss
        "global": torch.zeros(len(GLOBAL_ATTR_MAP), 4),
        "player": torch.zeros(len(PLAYER_ATTR_MAP), 4),
        "hex": torch.zeros(len(HEX_ATTR_MAP), 4)
    }

    for group in ["global", "player", "hex"]:
        _print("================================================================")
        # _print("==== Group: %s" % group)
        # train_loss = train_agglosses[group]
        # eval_loss = eval_agglosses[group]
        # diff_loss = eval_loss - train_loss
        # diff_frac = (diff_loss / train_loss) if train_loss != 0 else float("nan")
        # _print("  * diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (diff_loss, diff_frac, train_loss, eval_loss))

        for subtype in ["continuous", "cont_nullbit", "binaries", "categoricals", "thresholds"]:
            # _print("==== Subtype: %s" % subtype)
            # train_loss = train_agglosses[subtype]
            # eval_loss = eval_agglosses[subtype]
            # diff_loss = eval_loss - train_loss
            # diff_frac = (diff_loss / train_loss) if train_loss != 0 else float("nan")
            # _print("  * diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (diff_loss, diff_frac, train_loss, eval_loss))

            _print("==== Attrs: %s/%s" % (group, subtype))
            if len(model.obs_index.attr_ids[group][subtype]) == 0:
                continue

            train_loss = train_attrlosses[group][model.obs_index.attr_ids[group][subtype]].sum()
            eval_loss = eval_attrlosses[group][model.obs_index.attr_ids[group][subtype]].sum()
            diff_loss = eval_loss - train_loss
            diff_frac = (diff_loss / train_loss) if train_loss != 0 else float("nan")
            _print("  * diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (diff_loss, diff_frac, train_loss, eval_loss))

            train_loss = train_attrlosses[group][model.obs_index.attr_ids[group][subtype]]
            eval_loss = eval_attrlosses[group][model.obs_index.attr_ids[group][subtype]]
            for i, attr_id in enumerate(model.obs_index.attr_ids[group][subtype]):
                var_name = attrnames[group][attr_id]
                attr_train_loss = train_loss[i]
                attr_eval_loss = eval_loss[i]
                attr_diff_loss = attr_eval_loss - attr_train_loss
                attr_diff_frac = (attr_diff_loss / attr_train_loss) if attr_train_loss != 0 else float("nan")
                attrstats[group][attr_id][:] = torch.tensor([attr_diff_loss, attr_diff_frac, attr_eval_loss, attr_train_loss])
                _print("    %-30s diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (var_name, attr_diff_loss, attr_diff_frac, attr_train_loss, attr_eval_loss))

    res = {}
    for group in attrstats.keys():
        res[group] = []
        for pair in zip(attrnames[group], attrstats[group]):
            res[group].append(pair)

    topk_frac = {}
    topk_diff = {}

    for group in attrstats:
        topk_frac_ind = attrstats[group][:, 0].topk(topk).indices
        topk_diff_ind = attrstats[group][:, 1].topk(topk).indices
        topk_frac[group] = [(attrnames[group][i], attrstats[group][i]) for i in topk_frac_ind]
        topk_diff[group] = [(attrnames[group][i], attrstats[group][i]) for i in topk_diff_ind]

    return res, topk_frac, topk_diff
