#For creating the model
import bentoml
from sklearn import datasets
from sklearn.svm import SVC

iris=datasets.load_iris()
X,y=iris.data,iris.target

model=SVC(gamma="scale")
model.fit(X,y)

saved_model=bentoml.sklearn.save_model("iris_clf",model)
print(f"model saved :{saved_model}")


# It got created in your C drive
#C:\Users\zaynb\bentoml\models\iris_clf\xk2nlpevd2t73aco