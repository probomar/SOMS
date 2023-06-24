import tools as t
from setup import *


t.initialization()
for i in range(int(motion.shape[0]/step)):
    t.update_scene(i)
