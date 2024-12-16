# 23_1 copy

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
datasets = load_digits()      # feature_name 때문에
x = datasets.data
y = datasets.target

random_state1=1223
random_state2=1223

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, random_state=random_state1)

#2. 모델 구성
model1 = DecisionTreeClassifier(random_state=random_state2)
model2 = RandomForestClassifier(random_state=random_state2)
model3 = GradientBoostingClassifier(random_state=random_state2)
model4 = XGBClassifier(random_state=random_state2)

models = [model1, model2, model3, model4]

import matplotlib.pyplot as plt
import numpy as np

# print(model)

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model.__class__.__name__)

print('random_state :', random_state1, random_state2)
for i, model in enumerate(models):
    model.fit(x_train, y_train)
    print('==========', model.__class__.__name__, '==========')
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)
    plt.subplot(2, 2, i+1)
    plot_feature_importances_dataset(model)

plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.tight_layout()      # 간격 안겹치게
plt.show()

'''
random_state : 1223 1223
========== DecisionTreeClassifier ==========
acc : 0.8472222222222222
[0.         0.00886642 0.00544332 0.01521574 0.00567348 0.04850847
 0.         0.         0.         0.0155011  0.01128834 0.00283519
 0.01376369 0.01847882 0.00077323 0.         0.         0.00441178
 0.00988328 0.00842401 0.04104339 0.08596583 0.00077323 0.
 0.00103098 0.         0.07540495 0.05639249 0.04959762 0.01583598
 0.01060644 0.         0.         0.06354131 0.00462313 0.00139182
 0.07712915 0.03225891 0.01308039 0.         0.         0.00292356
 0.0745241  0.05873763 0.02066762 0.00885585 0.00535316 0.
 0.         0.00152153 0.00341824 0.00489287 0.00391772 0.01633694
 0.02674242 0.         0.         0.         0.00115985 0.00601834
 0.05943809 0.00201041 0.         0.00573917]
========== RandomForestClassifier ==========
acc : 0.9777777777777777
[0.00000000e+00 1.46684371e-03 2.15828252e-02 8.63100042e-03
 7.74184369e-03 2.11484828e-02 9.75321481e-03 6.78704204e-04
 9.99062839e-05 1.05410820e-02 2.10193342e-02 7.69191223e-03
 1.81855748e-02 2.60401050e-02 5.48039124e-03 5.48899644e-04
 3.24578083e-05 6.79716398e-03 2.34860951e-02 2.37832862e-02
 3.03543272e-02 5.15856136e-02 8.81822788e-03 2.96316379e-04
 2.55452052e-05 1.23990699e-02 4.38164383e-02 2.48623827e-02
 3.31805152e-02 2.20501425e-02 2.71615704e-02 4.56565143e-05
 0.00000000e+00 2.85024358e-02 2.85417748e-02 1.85314325e-02
 3.94845951e-02 2.02555900e-02 2.41710385e-02 0.00000000e+00
 2.82560661e-05 1.08591464e-02 3.44453029e-02 4.44882149e-02
 2.14025504e-02 1.74062193e-02 2.14496422e-02 1.37295796e-04
 1.23540702e-04 2.39008914e-03 1.72195928e-02 2.13446288e-02
 1.33219640e-02 2.63532308e-02 2.50370391e-02 1.21113019e-03
 0.00000000e+00 2.03549096e-03 2.50731385e-02 1.02211980e-02
 2.51334337e-02 2.97603141e-02 1.84862656e-02 3.28051993e-03]
========== GradientBoostingClassifier ==========
acc : 0.9638888888888889
[0.00000000e+00 9.43576218e-04 8.27093418e-03 2.99739028e-03
 2.97008861e-03 5.88988652e-02 5.62951522e-03 4.34895132e-04
 8.88357015e-04 1.95155677e-03 2.09891586e-02 6.91858455e-04
 9.24903709e-03 7.02342073e-03 1.07147802e-03 9.52195665e-04
 8.50885455e-05 3.76139301e-03 1.66091111e-02 3.89863704e-02
 1.91841797e-02 8.88369854e-02 6.61116977e-03 2.37200119e-07
 1.34856142e-04 9.39554807e-04 4.67291318e-02 1.70291837e-02
 3.59639663e-02 2.54396537e-02 1.23329945e-02 1.21625408e-04
 0.00000000e+00 5.92042227e-02 4.25109611e-03 4.85830972e-03
 7.20654432e-02 9.73968712e-03 1.60211041e-02 0.00000000e+00
 0.00000000e+00 7.10152985e-03 7.93040245e-02 7.13910682e-02
 9.18213134e-03 2.09073603e-02 2.88972048e-02 2.95059335e-04
 2.73206342e-06 1.10655416e-03 5.54005762e-03 1.59939737e-02
 7.94602878e-03 1.39723083e-02 2.80811998e-02 3.48132723e-04
 6.15332082e-04 8.99662610e-05 1.51837762e-02 5.91428210e-04
 5.62177231e-02 5.95652996e-03 2.14412261e-02 7.96696088e-03]
========== XGBClassifier ==========
acc : 0.9694444444444444
[0.         0.03260561 0.00599625 0.00737296 0.00518664 0.04148781
 0.00560644 0.         0.         0.00720591 0.0132025  0.00289933
 0.01232869 0.01128461 0.00088012 0.01497376 0.         0.00730367
 0.00837447 0.04088178 0.01003765 0.04985778 0.00374498 0.
 0.         0.00520189 0.03477967 0.01006788 0.03358783 0.02671578
 0.01016452 0.         0.         0.07282628 0.00650188 0.00796701
 0.05700776 0.01347518 0.03782811 0.         0.         0.0100892
 0.03535675 0.03744606 0.00969996 0.01911054 0.02964951 0.
 0.         0.00542436 0.00263066 0.01368503 0.01321433 0.01782674
 0.03249998 0.         0.         0.         0.02799876 0.00390022
 0.06468432 0.01331835 0.03346457 0.03264587]
'''
