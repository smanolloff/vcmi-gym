#!/bin/zsh

export PS4="+[%D{%Y-%m-%d %H:%M:%S}]<$$> "
set -x

[ -r rl/evaluate.py ] || { echo "File not found: rl/evaluate.py"; exit 1; }

while true; do
  python rl/evaluate.py
  sleep 30
done

