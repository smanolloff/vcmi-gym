#!/bin/bash

source ~/.simorc

set -euxo pipefail

# Usage:
#
# $0 [venvs] [vsteps]
#

players=(
    models/ihggsmxw-1782726477-model-dna.pt
    models/zvytfdpo-best27-model-ppo.pt
    models/zvytfdpo-best29-model-ppo.pt
)

opponents=(
    BattleAI
    mmai/models/attacker-nkjrmrsq-202509291846-stochastic.onnx
)

for player in "${players[@]}"; do
    [ -f $player ] || { echo "Not found: $player"; exit 1; }
done

for opponent in "${opponents[@]}"; do
    [ -f vcmi/Mods/$opponent -o $opponent = "BattleAI" ] || { echo "Not found: vcmi/Mods/$opponent"; exit 1; }
done

for player in "${players[@]}"; do
    for opponent in "${opponents[@]}"; do
        for town_chance in 0; do
            cat <<-EOF
=========================== $player vs. $opponent (town_chance=$town_chance)
EOF
            python -m vcmi_gym.tools.arena \
                --num-envs=${1-40} \
                --num-vsteps=${2-10000} \
                --player=$player \
                --opponent=$opponent \
                --map=gym/ml-eval.vmap \
                --envarg seed=42 \
                --envarg town_chance=$town_chance \
                --envarg warmachine_chance=20 \
                --envarg random_armies=False \
                --envarg random_heroes=1 \
                --envarg random_obstacles=1 \
                --envarg random_terrain_chance=100 \
                --envarg random_primary_skills=0
        done
    done
done
