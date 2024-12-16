import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)

outliers.fit(aaa)
results = outliers.predict(aaa)
# print(results)              # ValueError: Expected 2D array, got 1D array instead:

##################################################
import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
print(aaa.shape)            # (13,)
aaa = aaa.reshape(-1, 1)    # (13, 1)

from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.3)
outliers = EllipticEnvelope()

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)              # [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]
