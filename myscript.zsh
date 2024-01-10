
function handle_sigint() {
  echo "*** [🐕] SIGINT caught"
  set -x
  echo $$
  pgrep -g 0
  sleep 1
  exit 0
}



trap "handle_sigint" INT
set -x
#setopt monitor
#unsetopt HUP
ps -o pgid= -p $$
sleep 10 &
sleep 50
echo "baba"


