# Usage:
#   python timecalc.py NUM_WORKERS TIME_START TIME_END
#
# Example:
#   python timecalc.py 32 2025-04-05T18:03:52.716 2025-04-05T18:05:02.825
#
# NOTE:
#   TIME_START is the time of the first "Loading samples..." log
#   TIME_END is the time of the last "Loading samples..." log of the 2nd batch
#   i.e. for 10 workers, each batch has 10 such messages
#   => TIME_END corresponds to the 21st

import sys
from datetime import datetime

samples = int(sys.argv[1]) * 5000 * 2
t1 = datetime.fromisoformat(sys.argv[2])
t2 = datetime.fromisoformat(sys.argv[3])
duration = (t2 - t1).seconds

print("%d samples, %d seconds" % (samples, duration))
print("=> %.0f samples/s" % (samples / duration))
