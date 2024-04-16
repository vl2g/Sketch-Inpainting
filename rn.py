import random
import os


files = sorted(os.listdir('RESULT/final_inf/297_085/inp'))

random.seed(52)
sel_files = random.sample(files, 50)

o_dir = '/DATA/nakul/sketch/SketchInpDiffusion/RESULT/final_inf/297_085/inp'
o_pth = []

p_dir = '/DATA/nakul/sketch/Palette/experiments/final_inf'
p_pth = []

d_dir = '/DATA/nakul/sketch/deepfillv2/results'
d_pth = []
for f in sel_files:
    o_pth.append(f'{o_dir}/{f}')
    p_pth.append(f'{p_dir}/{f}')
    d_pth.append(f'{d_dir}/{f}')

for lit in ['o', 'p', 'd']:
    l = eval(f'{lit}_pth')
    with open(f'{lit}_1.txt', 'w') as f:
        for item in l:
            f.write(f'{item}\n')
    
    print(f'Done with {lit}')

