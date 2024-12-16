from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)     # (150, 4) (150,)

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

print('random_state :', random_state1, random_state2)
for model in models:
    model.fit(x_train, y_train)
    print('==========', model.__class__.__name__, '==========')
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)

'''
random_state : 123 7777
========== DecisionTreeClassifier ==========
acc : 0.8333333333333334
[0.         0.0425     0.42133357 0.53616643]
========== RandomForestClassifier ==========
acc : 0.9333333333333333
[0.1026199  0.02183504 0.48612713 0.38941793]
========== GradientBoostingClassifier ==========
acc : 0.9666666666666667
[0.00162142 0.02143047 0.68475052 0.2921976 ]
========== XGBClassifier ==========
acc : 0.9333333333333333
[0.02430454 0.02472077 0.7376847  0.21328996]

random_state : 1223 1223
========== DecisionTreeClassifier ==========
acc : 1.0
[0.01666667 0.         0.57742557 0.40590776]
========== RandomForestClassifier ==========
acc : 1.0
[0.10691492 0.02814393 0.42049394 0.44444721]
========== GradientBoostingClassifier ==========
acc : 1.0
[0.01074646 0.01084882 0.27282247 0.70558224]
========== XGBClassifier ==========
acc : 1.0
[0.00897023 0.02282782 0.6855639  0.28263798]
'''

# [검색] 메시지 없애기 : model.__class__.__name__
'''
random_state : 1223
========== DecisionTreeClassifier(random_state=1223) ==========
acc : 1.0
[0.01666667 0.         0.57742557 0.40590776]
========== RandomForestClassifier(random_state=1223) ==========
acc : 1.0
[0.10691492 0.02814393 0.42049394 0.44444721]
========== GradientBoostingClassifier(random_state=1223) ==========
acc : 1.0
[0.01074646 0.01084882 0.27282247 0.70558224]
========== XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=None,
              num_parallel_tree=None, objective='multi:softprob', ...) ==========
acc : 1.0
[0.00897023 0.02282782 0.6855639  0.28263798]
'''
