import numpy as np
import math
import os
import pyvista as pv
import pandas as pd

# model = 'simple'
model = 'full'

z_step = 0.02
fiy_step = 0.05
step = 10

E = 10.35  # MPa

kACL = 10
kPCL = 10
kLCL = 10
kMCL = 10

ACL0 = 16
PCL0 = 30
LCL0 = 50
MCL0 = 30

file0 = 'cor0.txt'
if os.path.exists(file0):
    os.remove(file0)

file = 'cor_lig_v3.csv'

if os.path.exists(file):
    os.remove(file)

file1 = 'F_M.csv'

if os.path.exists(file1):
    os.remove(file1)

pv.global_theme.show_edges = True

if model == 'full':
    femur = pv.read('Models/Segmentation_Model_95_femur.stl')
    cartilage = pv.read('Models/Segmentation_Model_84_lateral_tibial_cartilage.stl') + \
                pv.read('Models/Segmentation_Model_85_medial_tibial_cartilage.stl')
    tibia = pv.read('Models/Segmentation_Model_96_tibia.stl')
    femoral_cartilage = pv.read('Models/Segmentation_Model_83_femoral_cartilage.stl')
    tibial_cartilage = pv.read('Models/Segmentation_Model_84_lateral_tibial_cartilage.stl') + \
                       pv.read('Models/Segmentation_Model_85_medial_tibial_cartilage.stl')

elif model == 'simple':
    femur = pv.read('Models_simply/Segmentation_Model_95_femur.stl')
    cartilage = pv.read('Models_simply/Segmentation_Model_84_lateral_tibial_cartilage.stl') + \
                pv.read('Models_simply/Segmentation_Model_85_medial_tibial_cartilage.stl')
    tibia = pv.read('Models_simply/Segmentation_Model_96_tibia.stl')
    femoral_cartilage = pv.read('Models_simply/Segmentation_Model_83_femoral_cartilage.stl')
    tibial_cartilage = pv.read('Models_simply/Segmentation_Model_84_lateral_tibial_cartilage.stl') + \
                       pv.read('Models_simply/Segmentation_Model_85_medial_tibial_cartilage.stl')

flex = femur
flex_cartilage = femoral_cartilage
full_flex = flex + flex_cartilage
full_tibia = tibia + tibial_cartilage

dfmot = pd.read_csv('my_motion.csv', sep=';')
dfmus = pd.read_csv('my_muscle_force.csv', sep=';')
motion = dfmot.to_numpy()
muscle = dfmus.to_numpy()
