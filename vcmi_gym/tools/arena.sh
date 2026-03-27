#!/bin/bash

source ~/.simorc

set -euxo pipefail

# Usage:
#
# $0 [venvs] [vsteps]
#

players=(
    tukbajrv-202509241418
    tukbajrv-1770544743
    idiqvwea-1773428519
    naesumvw-best2
    tukbavip-1773500887
    fqcbvmti-best5
)

opponents=(
    BattleAI
    nkjrmrsq-202509291846
)

for player in "${players[@]}"; do
    player_file=data/mppo-dna-heads/$player-model-dna.pt
    [ -f $player_file ] || download_model $player
    for opponent in "${opponents[@]}"; do
        if [ $opponent = BattleAI ]; then
            opponent_file=$opponent
        else
            opponent_file=data/mppo-dna-heads/$opponent-model-dna.pt
            [ -f $opponent_file ] || download_model $opponent
        fi

        # Prevent cleanup.sh cron from deleting the old checkpoints
        touch data/mppo-dna-heads/*

        for town_chance in 0 100; do
            cat <<-EOF
=========================== $player vs. $opponent (town_chance=$town_chance)
EOF
            python -m vcmi_gym.tools.arena \
                --num-envs=${1-30} \
                --num-vsteps=${2-1000} \
                --player=$player_file \
                --opponent=$opponent_file \
                --envarg town_chance=$town_chance
        done
    done
done
