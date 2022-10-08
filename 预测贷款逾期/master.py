#开发人：YE
#开发时间： 22:14
import pandas
from sklearn.model_selection import train_test_split
from  sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#数据集读取
model=pandas.read_csv("model.csv")#训练集
test=pandas.read_csv("test.csv")#测试集
pandas.set_option('display.max_rows', None)
model.isnull().sum()#查看各列缺失值个数

#删除没用或者缺失值较多列
model=model.drop(['user_id'],axis=1)
test=test.drop(['user_id'],axis=1)
#定义函数计算列缺失率并删除缺失率高于指定值的列
def drop_col(df, col_name, cutoff):
    n = len(df)
    cnt = df[col_name].count()
    if (float(cnt) / n) < cutoff:
        df.drop(col_name, axis=1, inplace=True)
        test.drop(col_name, axis=1, inplace=True)
#遍历训练集调用上面函数
for i in model:
    drop_col(model,i,0.01)
#剩下的缺失值用中位数填充
model.iloc[:,1:]=model.iloc[:,1:].fillna(model.iloc[:,1:].median())
test=test.fillna(test.median())
#划分特征和标签
modely=model['y']
modelx=model.drop(['y'],axis=1)
#划分数据集
X_train, X_test, y_train, y_test=train_test_split(modelx,modely,test_size=0.2,random_state=135)

#数据归一化
from sklearn.preprocessing import minmax_scale
  # 归一化，缩放到0-1
X_train = minmax_scale(X_train)
X_test =  minmax_scale(X_test)


# 模型训练
linearSVC = LinearSVC(dual=False)
linearSVC.fit(X_train, y_train)
linearSVC_predict = linearSVC.predict(test)
for i in linearSVC_predict:
    print(i,end=' ')
size=sum(linearSVC_predict[linearSVC_predict==1])
print(size)
#输出结果
print("predict:",linearSVC.score(X_test, y_test))
