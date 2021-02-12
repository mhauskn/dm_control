import sys
import pstats

p = pstats.Stats(sys.argv[1])
p.sort_stats('cumtime').print_stats(50)