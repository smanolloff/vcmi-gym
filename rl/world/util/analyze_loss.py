import torch

from ..util.constants_v12 import GLOBAL_ATTR_MAP, PLAYER_ATTR_MAP, HEX_ATTR_MAP

from ..util.obs_index import ContextGroup, DataGroup


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

    attrnames = {
        ContextGroup.GLOBAL: list(GLOBAL_ATTR_MAP),
        ContextGroup.PLAYER: list(PLAYER_ATTR_MAP),
        ContextGroup.HEX: list(HEX_ATTR_MAP)
    }

    attrstats = {
        # 4 = diff_frac, diff_loss, eval_loss, train_loss
        ContextGroup.GLOBAL: torch.zeros(len(GLOBAL_ATTR_MAP), 4),
        ContextGroup.PLAYER: torch.zeros(len(PLAYER_ATTR_MAP), 4),
        ContextGroup.HEX: torch.zeros(len(HEX_ATTR_MAP), 4)
    }

    for context in ContextGroup.as_list():
        _print("================================================================")
        # _print("==== Group: %s" % context)
        # train_loss = train_agglosses[context]
        # eval_loss = eval_agglosses[context]
        # diff_loss = eval_loss - train_loss
        # diff_frac = (diff_loss / train_loss) if train_loss != 0 else float("nan")
        # _print("  * diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (diff_loss, diff_frac, train_loss, eval_loss))

        for datatype in DataGroup.as_list():
            # _print("==== Subtype: %s" % datatype)
            # train_loss = train_agglosses[datatype]
            # eval_loss = eval_agglosses[datatype]
            # diff_loss = eval_loss - train_loss
            # diff_frac = (diff_loss / train_loss) if train_loss != 0 else float("nan")
            # _print("  * diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (diff_loss, diff_frac, train_loss, eval_loss))

            _print("==== Attrs: %s/%s" % (context, datatype))
            if len(model.obs_index.attr_ids[context][datatype]) == 0:
                continue

            train_loss = train_attrlosses[context][model.obs_index.attr_ids[context][datatype]].sum()
            eval_loss = eval_attrlosses[context][model.obs_index.attr_ids[context][datatype]].sum()
            diff_loss = eval_loss - train_loss
            diff_frac = (diff_loss / train_loss) if train_loss != 0 else float("nan")
            _print("  * diff: %.3f (x%.1f) | train: %.3f | eval: %.3f" % (diff_loss, diff_frac, train_loss, eval_loss))

            train_loss = train_attrlosses[context][model.obs_index.attr_ids[context][datatype]]
            eval_loss = eval_attrlosses[context][model.obs_index.attr_ids[context][datatype]]
            for i, attr_id in enumerate(model.obs_index.attr_ids[context][datatype]):
                var_name = attrnames[context][attr_id]
                attr_train_loss = train_loss[i]
                attr_eval_loss = eval_loss[i]
                attr_diff_loss = attr_eval_loss - attr_train_loss
                attr_diff_frac = (attr_diff_loss / attr_train_loss) if attr_train_loss != 0 else float("nan")
                attrstats[context][attr_id][:] = torch.tensor([attr_diff_loss, attr_diff_frac, attr_eval_loss, attr_train_loss])
                _print("    %-30s diff: %.3f (x%.1f) | eval: %.3f | train: %.3f" % (var_name, attr_diff_loss, attr_diff_frac, attr_eval_loss, attr_train_loss))

    res = {}
    for context in attrstats.keys():
        res[context] = list(zip(attrnames[context], attrstats[context]))

    topk_frac = {}
    topk_diff = {}

    for context in attrstats:
        k = min(topk, attrstats[context].shape[0])
        if k == 0:
            continue
        topk_frac_ind = attrstats[context][:, 0].topk(k).indices
        topk_diff_ind = attrstats[context][:, 1].topk(k).indices
        topk_frac[context] = [(attrnames[context][i], attrstats[context][i]) for i in topk_frac_ind]
        topk_diff[context] = [(attrnames[context][i], attrstats[context][i]) for i in topk_diff_ind]

    return res, topk_frac, topk_diff
