import numpy as np

#cube1 = np.array([])
#for i in range (100):
#	cube2 = np.load(f'Z_cube_AAA_{i}.npy')
#	cube1 = np.concatenate((cube1,cube2))


cube = np.vstack([np.load(f'Z_cube_AAA_{i}.npy') for i in range(100)])

np.save('Z_cube_2024.npy', cube)
