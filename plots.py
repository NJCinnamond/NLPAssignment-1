import numpy as np 
import matplotlib.pyplot as pyplot

nomask_losses = [1.181096193583116,0.5412596282751663,0.5168246244606765,0.5052332385726596,0.5002813002337579,0.49267008187978173,0.49703964655813976,0.49006101813005365,0.48472113194672956,0.4820000976324081]

pyplot.figure()
pyplot.plot(range(len(nomask_losses)),nomask_losses)
pyplot.xlabel("Number of Epochs")
pyplot.ylabel("Cross-Entropy Loss")
pyplot.show()